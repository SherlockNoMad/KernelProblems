import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    freq_ptr,
    idx_ptr,
    out_ptr,
    T,
    BLOCK_T: tl.constexpr,
    H_CONST: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    freq_idx = tl.load(idx_ptr + t_offs, mask=t_mask, other=0).to(tl.int32)

    pair_offs = tl.arange(0, 64)

    freq_base = freq_idx[:, None] * 128 + pair_offs[None, :] * 2
    fr = tl.load(freq_ptr + freq_base, mask=t_mask[:, None], other=0.0)
    fi = tl.load(freq_ptr + freq_base + 1, mask=t_mask[:, None], other=0.0)

    head_base = pid_h * 128
    re_offs = t_offs[:, None] * (H_CONST * 128) + head_base + 2 * pair_offs[None, :]
    im_offs = re_offs + 1

    xr = tl.load(x_ptr + re_offs, mask=t_mask[:, None], other=0.0).to(tl.float32)
    xi = tl.load(x_ptr + im_offs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    out_re = xr * fr - xi * fi
    out_im = xr * fi + xi * fr

    tl.store(out_ptr + re_offs, out_re.to(out_ptr.dtype.element_ty), mask=t_mask[:, None])
    tl.store(out_ptr + im_offs, out_im.to(out_ptr.dtype.element_ty), mask=t_mask[:, None])


def kernel_function(mm, mm_1, arg586_1, arg582_1):
    T = 8192

    freq_real_view = torch.view_as_real(arg582_1)
    idx = arg586_1.view(-1)

    out_q = torch.empty((1, 8192, 32, 128), dtype=torch.bfloat16, device=mm.device)
    out_k = torch.empty((1, 8192, 8, 128), dtype=torch.bfloat16, device=mm.device)

    BLOCK_T_Q = 8
    grid_q = (triton.cdiv(T, BLOCK_T_Q), 32)
    _rope_kernel[grid_q](
        mm, freq_real_view, idx, out_q,
        T,
        BLOCK_T=BLOCK_T_Q,
        H_CONST=32,
        num_warps=4,
        num_stages=2,
    )

    BLOCK_T_K = 8
    grid_k = (triton.cdiv(T, BLOCK_T_K), 8)
    _rope_kernel[grid_k](
        mm_1, freq_real_view, idx, out_k,
        T,
        BLOCK_T=BLOCK_T_K,
        H_CONST=8,
        num_warps=4,
        num_stages=2,
    )

    return (out_q, out_k)