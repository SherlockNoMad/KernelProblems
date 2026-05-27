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
      a = mm_4, b = mm_5, c = mm_662
      sig_a = sigmoid(a)
      silu_a = a * sig_a
      out0 = silu_a * b                                    (mul_tensor)
      out1 = c * silu_a                                    (mul_tensor_1)
      out2 = silu_backward(c * b, a)
           = (c*b) * sig_a * (1 + a * (1 - sig_a))
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    a = tl.load(mm4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(mm662_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a

    out0 = silu_a * b
    out1 = c * silu_a
    cb = c * b
    out2 = cb * sig_a * (1.0 + a * (1.0 - sig_a))

    tl.store(out0_ptr + offs, out0.to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + offs, out1.to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + offs, out2.to(tl.bfloat16), mask=mask)


def kernel_function(mm_4, mm_5, mm_662):
    assert mm_4.is_cuda and mm_5.is_cuda and mm_662.is_cuda
    assert mm_4.shape == mm_5.shape == mm_662.shape
    assert mm_4.dtype == mm_5.dtype == mm_662.dtype

    mm_4_c = mm_4.contiguous()
    mm_5_c = mm_5.contiguous()
    mm_662_c = mm_662.contiguous()

    out0 = torch.empty_like(mm_4_c)
    out1 = torch.empty_like(mm_4_c)
    out2 = torch.empty_like(mm_4_c)

    n_elements = mm_4_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_kernel[grid](
        mm_4_c, mm_5_c, mm_662_c,
        out0, out1, out2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out0, out1, out2)