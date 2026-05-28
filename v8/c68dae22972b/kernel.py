import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SiLU(a) * b kernel.
    
    Fused stages:
      1. Load a, b
      2. Compute silu(a) = a * sigmoid(a) in fp32
      3. Multiply by b
      4. Cast back and store
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sig
    result = silu * b

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_4, mm_5):
    """Fused SiLU * Mul wrapper.
    
    Performs: out = silu(mm_4) * mm_5 in a single Triton kernel pass.
    Wrapper only allocates output and launches the kernel.
    """
    assert mm_4.is_cuda and mm_5.is_cuda
    assert mm_4.shape == mm_5.shape
    assert mm_4.dtype == mm_5.dtype

    out = torch.empty_like(mm_4)
    n_elements = mm_4.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Ensure contiguous flat view for elementwise op
    a = mm_4.contiguous()
    b = mm_5.contiguous()

    _silu_mul_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out