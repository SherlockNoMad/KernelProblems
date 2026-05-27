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
      silu = silu(mm4) = mm4 * sigmoid(mm4)
      out0 = silu * mm5                          (view_default = mul_tensor)
      out1 = mm662 * silu                        (view_default_2 = mul_tensor_1)
      tmp  = mm662 * mm5                         (mul_tensor_2)
      out2 = silu_backward(tmp, mm4)             (view_default_3)
        silu_backward(grad, x) = grad * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(mm4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(mm5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(mm662_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-x))
    silu = x * sig

    out0 = silu * y
    out1 = g * silu
    grad_in = g * y  # mul_tensor_2 = mm662 * mm5
    out2 = grad_in * sig * (1.0 + x * (1.0 - sig))

    tl.store(out0_ptr + offsets, out0.to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + offsets, out1.to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + offsets, out2.to(tl.bfloat16), mask=mask)


def kernel_function(mm_4, mm_5, mm_662):
    """
    Fused implementation of the Model's forward pass.

    Single fused kernel performing:
      1. silu(mm_4)
      2. out0 = silu * mm_5                  (view_default)
      3. out1 = mm_662 * silu                (view_default_2)
      4. tmp  = mm_662 * mm_5                (mul_tensor_2)
      5. out2 = silu_backward(tmp, mm_4)     (view_default_3)
    """
    assert mm_4.is_cuda and mm_5.is_cuda and mm_662.is_cuda
    assert mm_4.shape == mm_5.shape == mm_662.shape
    assert mm_4.dtype == torch.bfloat16

    mm_4_c = mm_4.contiguous()
    mm_5_c = mm_5.contiguous()
    mm_662_c = mm_662.contiguous()

    out_shape = (8192, 14336)
    out0 = torch.empty(out_shape, dtype=torch.bfloat16, device=mm_4.device)
    out1 = torch.empty(out_shape, dtype=torch.bfloat16, device=mm_4.device)
    out2 = torch.empty(out_shape, dtype=torch.bfloat16, device=mm_4.device)

    n_elements = out0.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_kernel[grid](
        mm_4_c, mm_5_c, mm_662_c,
        out0, out1, out2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return (out0, out1, out2)