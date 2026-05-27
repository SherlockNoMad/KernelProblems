import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SiLU(a) * b kernel. SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu = a * (1.0 / (1.0 + tl.exp(-a)))
    result = silu * b

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(a, b):
    """
    Fused kernel computing: SiLU(a) * b elementwise.
    Fuses: sigmoid -> mul (silu) -> mul (with b) into a single pass.
    """
    assert a.shape == b.shape, "Input shapes must match"
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.is_cuda, "Inputs must be CUDA tensors"

    a_contig = a.contiguous()
    b_contig = b.contiguous()
    output = torch.empty_like(a_contig)
    n_elements = a_contig.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_kernel[grid](a_contig, b_contig, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output