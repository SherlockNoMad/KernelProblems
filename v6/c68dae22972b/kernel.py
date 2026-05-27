import triton
import triton.language as tl
import torch


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SiLU(a) * b elementwise kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_a = a * (1.0 / (1.0 + tl.exp(-a)))
    result = silu_a * b

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_221, mm_222):
    """
    Fused SiLU + multiply kernel.
    
    Fused stages:
      1. SiLU(mm_221) = mm_221 * sigmoid(mm_221)
      2. Element-wise multiply with mm_222
    
    Both stages execute in a single Triton kernel pass.
    """
    assert mm_221.shape == mm_222.shape, "Input shapes must match"
    assert mm_221.dtype == mm_222.dtype, "Input dtypes must match"
    assert mm_221.is_cuda and mm_222.is_cuda, "Inputs must be on CUDA"

    # Ensure contiguous for flat indexing
    a = mm_221.contiguous()
    b = mm_222.contiguous()

    output = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output