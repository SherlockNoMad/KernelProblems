import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    mm4_ptr, mm5_ptr, mm662_ptr,
    out0_ptr, out1_ptr, out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel computing:
      s = silu(mm4) = mm4 * sigmoid(mm4)
      out0 = s * mm5                          (view_default)
      out1_flat = mm662 * s                   (will be transposed in wrapper view)
      out2_flat = silu_backward(mm662 * mm5, mm4)
                = (mm662 * mm5) * sigmoid(mm4) * (1 + mm4 * (1 - sigmoid(mm4)))
    
    All three outputs are written in a single pass over the flattened tensor.
    Transposes are handled as views in the wrapper (no compute needed beyond
    storing into the transposed layout's underlying contiguous buffer).
    
    Actually: t_default on [8192, 14336] returns a view with shape [14336, 8192].
    To keep things simple and match reference output exactly (including stride),
    we store the [8192, 14336] result and call .t() in the wrapper - but .t()
    is just a view (no compute), so it's allowed.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(mm4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(mm662_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig_a = 1.0 / (1.0 + tl.exp(-a))
    silu_a = a * sig_a

    # out0 = silu(a) * b
    out0 = silu_a * b

    # out1_pre_t = c * silu(a)  (before transpose)
    out1 = c * silu_a

    # silu_backward(grad_out=c*b, self=a)
    # d_silu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    grad = c * b
    d_silu = sig_a * (1.0 + a * (1.0 - sig_a))
    out2 = grad * d_silu

    tl.store(out0_ptr + offsets, out0.to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + offsets, out1.to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + offsets, out2.to(tl.bfloat16), mask=mask)


def kernel_function(mm_4, mm_5, mm_662):
    """
    Fused implementation of the model:
      - view_default = silu(mm_4) * mm_5    [8192, 14336]
      - t_default    = (mm_662 * silu(mm_4)).t()  [14336, 8192]
      - t_default_1  = silu_backward(mm_662 * mm_5, mm_4).t()  [14336, 8192]
    
    All elementwise math is fused into a single Triton kernel pass.
    Transposes are zero-compute view operations.
    """
    assert mm_4.is_cuda and mm_5.is_cuda and mm_662.is_cuda
    assert mm_4.shape == mm_5.shape == mm_662.shape
    assert mm_4.dtype == torch.bfloat16

    # Ensure contiguous
    mm_4 = mm_4.contiguous()
    mm_5 = mm_5.contiguous()
    mm_662 = mm_662.contiguous()

    M, N = mm_4.shape  # 8192, 14336
    n_elements = mm_4.numel()

    out0 = torch.empty((M, N), dtype=torch.bfloat16, device=mm_4.device)
    out1_pre = torch.empty((M, N), dtype=torch.bfloat16, device=mm_4.device)
    out2_pre = torch.empty((M, N), dtype=torch.bfloat16, device=mm_4.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_kernel[grid](
        mm_4, mm_5, mm_662,
        out0, out1_pre, out2_pre,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    # Transposes are view ops (no compute)
    out1 = out1_pre.t()
    out2 = out2_pre.t()

    return (out0, out1, out2)