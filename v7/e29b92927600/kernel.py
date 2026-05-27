import torch
import triton
import triton.language as tl

# Fused kernel:
# - reshape mm_218 [8192, 1024] -> [1, 8192, 8, 64, 2] (last dim = real/imag pairs)
# - convert bf16 -> fp32
# - complex multiply with view_788 [1, 8192, 1, 64] (broadcast over heads dim=8)
#     (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
# - convert result back to bf16
# - output shape [1, 8192, 8, 128] (interleaved real,imag)
#
# Total elements = 1 * 8192 * 8 * 64 = 4,194,304 complex pairs
# Each program handles BLOCK pairs.

@triton.jit
def _rope_complex_mul_kernel(
    x_ptr,        # bf16, shape [8192, 1024] contiguous
    freq_ptr,     # complex64 viewed as fp32 pairs, shape [1, 8192, 1, 64, 2]
    out_ptr,      # bf16, shape [1, 8192, 8, 128]
    n_pairs,      # total number of complex pairs = 8192 * 8 * 64
    H: tl.constexpr,   # 8
    D: tl.constexpr,   # 64 (half of head_dim)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs

    # Decompose linear index into (seq, head, d)
    # layout: seq * (H*D) + head * D + d
    d = offs % D
    tmp = offs // D
    head = tmp % H
    seq = tmp // H

    # Input layout: mm_218 is [8192, 1024], viewed as [1, 8192, 8, 64, 2]
    # which is [seq, 1024] where 1024 = H*D*2 = 8*64*2
    # element index in flat input = seq * 1024 + head * (D*2) + d * 2 + {0,1}
    x_base = seq * 1024 + head * (D * 2) + d * 2
    a = tl.load(x_ptr + x_base + 0, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + x_base + 1, mask=mask, other=0.0).to(tl.float32)

    # Frequency layout: [1, 8192, 1, 64] complex, viewed as pairs:
    # flat real index = seq * (D*2) + d * 2
    f_base = seq * (D * 2) + d * 2
    c = tl.load(freq_ptr + f_base + 0, mask=mask, other=0.0)
    e = tl.load(freq_ptr + f_base + 1, mask=mask, other=0.0)

    # complex multiply
    real = a * c - b * e
    imag = a * e + b * c

    # Output layout: [1, 8192, 8, 128] contiguous
    # flat index = seq * (H * D * 2) + head * (D * 2) + d * 2 + {0,1}
    out_base = seq * (H * D * 2) + head * (D * 2) + d * 2
    tl.store(out_ptr + out_base + 0, real.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + 1, imag.to(tl.bfloat16), mask=mask)


def kernel_function(mm_218, view_788):
    """
    Fused RoPE-style complex multiply.

    Stages fused into one Triton kernel:
      1. Reshape mm_218 (bf16 [8192,1024]) as [1,8192,8,64,2]
      2. Cast to fp32
      3. View as complex (interpret last-dim pairs as real/imag)
      4. Complex multiply with view_788 (broadcast across heads dim)
      5. View as real (back to [...,2])
      6. Reshape to [1,8192,8,128]
      7. Cast to bf16
    """
    assert mm_218.is_cuda and view_788.is_cuda
    assert mm_218.dtype == torch.bfloat16
    assert view_788.dtype == torch.complex64
    assert mm_218.shape == (8192, 1024)
    assert view_788.shape == (1, 8192, 1, 64)

    mm_218_c = mm_218.contiguous()
    view_788_c = view_788.contiguous()

    # View complex64 tensor as fp32 with last dim=2 (real, imag)
    freq_real = torch.view_as_real(view_788_c)  # [1, 8192, 1, 64, 2] float32
    freq_real = freq_real.contiguous()

    out = torch.empty((1, 8192, 8, 128), dtype=torch.bfloat16, device=mm_218.device)

    H = 8
    D = 64
    n_pairs = 8192 * H * D  # 4,194,304

    BLOCK = 1024
    grid = (triton.cdiv(n_pairs, BLOCK),)

    _rope_complex_mul_kernel[grid](
        mm_218_c, freq_real, out,
        n_pairs,
        H=H, D=D, BLOCK=BLOCK,
    )
    return out