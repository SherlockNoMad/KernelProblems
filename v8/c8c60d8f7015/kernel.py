import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    idx_ptr,          # int32, [S]
    freqs_ptr,        # complex64 viewed as float32 [N, 64, 2]
    in_ptr,           # bf16 [S, H*128]
    out_ptr,          # bf16 [S, H, 128]
    S: tl.constexpr,
    H: tl.constexpr,
    F: tl.constexpr,  # 64
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid encodes (s, h)
    s = pid // H
    h = pid % H

    # Load position index
    idx = tl.load(idx_ptr + s).to(tl.int64)

    offs_f = tl.arange(0, BLOCK_F)
    mask = offs_f < F

    # Load freqs (real, imag) for row idx
    # freqs_ptr layout: [N, F, 2] as float32
    freq_base = idx * (F * 2)
    c = tl.load(freqs_ptr + freq_base + offs_f * 2, mask=mask, other=0.0)       # real
    d = tl.load(freqs_ptr + freq_base + offs_f * 2 + 1, mask=mask, other=0.0)   # imag

    # Load input (bf16) for (s, h, 2f) and (s, h, 2f+1)
    in_base = s * (H * F * 2) + h * (F * 2)
    a = tl.load(in_ptr + in_base + offs_f * 2, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(in_ptr + in_base + offs_f * 2 + 1, mask=mask, other=0.0).to(tl.float32)

    # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    real_out = a * c - b * d
    imag_out = a * d + b * c

    # Store output (bf16) to [s, h, 2f] and [s, h, 2f+1]
    out_base = s * (H * F * 2) + h * (F * 2)
    tl.store(out_ptr + out_base + offs_f * 2, real_out.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + offs_f * 2 + 1, imag_out.to(tl.bfloat16), mask=mask)


def kernel_function(arg586_1, arg582_1, mm, mm_1):
    """
    Fused RoPE-style rotary embedding application.

    Stages fused inside the Triton kernel (per output element):
      1. Gather complex rotary frequency from arg582_1 using index from arg586_1.
      2. Cast bf16 input pair (real, imag) to fp32.
      3. Complex multiply with frequency.
      4. Cast back to bf16 and store.

    Two kernel launches (Q and K) share the same kernel since they differ only in H.
    """
    assert arg586_1.is_cuda and arg582_1.is_cuda and mm.is_cuda and mm_1.is_cuda
    assert arg586_1.shape == (1, 8192)
    assert arg582_1.shape == (8192, 64) and arg582_1.dtype == torch.complex64
    assert mm.shape == (8192, 4096) and mm.dtype == torch.bfloat16
    assert mm_1.shape == (8192, 1024) and mm_1.dtype == torch.bfloat16

    device = arg586_1.device
    S = 8192
    F = 64

    # Allocate outputs
    out_q = torch.empty((1, S, 32, 128), dtype=torch.bfloat16, device=device)
    out_k = torch.empty((1, S, 8, 128), dtype=torch.bfloat16, device=device)

    # View complex tensor as float32 for direct indexing in the kernel
    freqs_f32 = torch.view_as_real(arg582_1).contiguous()  # [8192, 64, 2] float32

    # Ensure contiguity
    idx_flat = arg586_1.view(-1).contiguous()
    mm_c = mm.contiguous()
    mm1_c = mm_1.contiguous()

    BLOCK_F = 64

    # Q: H = 32
    grid_q = (S * 32,)
    _rope_kernel[grid_q](
        idx_flat, freqs_f32, mm_c, out_q,
        S=S, H=32, F=F, BLOCK_F=BLOCK_F,
    )

    # K: H = 8
    grid_k = (S * 8,)
    _rope_kernel[grid_k](
        idx_flat, freqs_f32, mm1_c, out_k,
        S=S, H=8, F=F, BLOCK_F=BLOCK_F,
    )

    return (out_q, out_k)