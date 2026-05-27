import torch
import triton
import triton.language as tl

# Fused RoPE-style kernel:
# - Reshape mm [8192,4096] -> [1,8192,32,128] and mm_1 [8192,1024] -> [1,8192,8,128]
# - Treat last dim as 64 complex pairs (real, imag)
# - Cast to fp32
# - Lookup freqs: arg582_1[arg586_1[0, t]] for each token t -> shape [8192, 64] complex
# - Complex multiply per (token, head, pair)
# - Store as bf16 [1, 8192, H, 128]


@triton.jit
def _rope_kernel(
    x_ptr,         # bf16 input [T, H*128] (T=8192)
    freq_re_ptr,   # fp32 freq real [F, 64] (F=8192)
    freq_im_ptr,   # fp32 freq imag [F, 64]
    idx_ptr,       # int32 indices [T]
    out_ptr,       # bf16 output [T, H*128]
    T, H,
    BLOCK_T: tl.constexpr,
    H_CONST: tl.constexpr,  # heads per token (compile time)
):
    pid_t = tl.program_id(0)  # which token block
    pid_h = tl.program_id(1)  # which head

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_mask = t_offs < T

    # Load freq indices per token
    freq_idx = tl.load(idx_ptr + t_offs, mask=t_mask, other=0).to(tl.int32)

    # pair index in last dim (0..63)
    pair_offs = tl.arange(0, 64)  # [64]

    # Load freq[freq_idx, pair] -> shape [BLOCK_T, 64]
    freq_ptr_off = freq_idx[:, None] * 64 + pair_offs[None, :]
    fr = tl.load(freq_re_ptr + freq_ptr_off, mask=t_mask[:, None], other=0.0)
    fi = tl.load(freq_im_ptr + freq_ptr_off, mask=t_mask[:, None], other=0.0)

    # Load x for this head: x[t, pid_h*128 + 2*pair], x[t, pid_h*128 + 2*pair+1]
    head_base = pid_h * 128
    # shape [BLOCK_T, 64]
    re_offs = t_offs[:, None] * (H_CONST * 128) + head_base + 2 * pair_offs[None, :]
    im_offs = re_offs + 1

    xr = tl.load(x_ptr + re_offs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    xi = tl.load(x_ptr + im_offs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    # complex mul: (xr + i*xi) * (fr + i*fi) = (xr*fr - xi*fi) + i*(xr*fi + xi*fr)
    out_re = xr * fr - xi * fi
    out_im = xr * fi + xi * fr

    tl.store(out_ptr + re_offs, out_re.to(out_ptr.dtype.element_ty), mask=t_mask[:, None])
    tl.store(out_ptr + im_offs, out_im.to(out_ptr.dtype.element_ty), mask=t_mask[:, None])


def kernel_function(mm, mm_1, arg586_1, arg582_1):
    """
    Fused RoPE-style transform.

    Stages fused into a single kernel per output tensor:
      1. View/reshape (handled implicitly via index arithmetic)
      2. Cast bf16 -> fp32
      3. Complex multiply with frequency table indexed by arg586_1
      4. Cast fp32 -> bf16 and store

    The two outputs (q with 32 heads, k with 8 heads) are produced by two
    launches of the same kernel because they have different head counts.
    """
    assert mm.is_cuda and mm_1.is_cuda
    assert mm.shape == (8192, 4096)
    assert mm_1.shape == (8192, 1024)
    assert arg586_1.shape == (1, 8192)
    assert arg582_1.shape == (8192, 64)
    assert arg582_1.dtype == torch.complex64

    T = 8192

    # View complex64 freq table as real [8192, 64, 2] (allocation/view only, no compute)
    freq_real_view = torch.view_as_real(arg582_1)  # [8192, 64, 2] fp32
    freq_re = freq_real_view[..., 0].contiguous()
    freq_im = freq_real_view[..., 1].contiguous()

    idx = arg586_1.view(-1).contiguous().to(torch.int32)

    # Outputs
    out_q = torch.empty((1, 8192, 32, 128), dtype=torch.bfloat16, device=mm.device)
    out_k = torch.empty((1, 8192, 8, 128), dtype=torch.bfloat16, device=mm.device)

    BLOCK_T = 32

    # Launch for q: H=32
    grid_q = (triton.cdiv(T, BLOCK_T), 32)
    _rope_kernel[grid_q](
        mm, freq_re, freq_im, idx, out_q,
        T, 32,
        BLOCK_T=BLOCK_T,
        H_CONST=32,
    )

    # Launch for k: H=8
    grid_k = (triton.cdiv(T, BLOCK_T), 8)
    _rope_kernel[grid_k](
        mm_1, freq_re, freq_im, idx, out_k,
        T, 8,
        BLOCK_T=BLOCK_T,
        H_CONST=8,
    )

    return (out_q, out_k)