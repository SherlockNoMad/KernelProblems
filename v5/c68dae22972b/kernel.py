import triton
import triton.language as tl
import torch


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused SiLU(a) * b kernel. All compute done in Triton."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # silu(a) = a * sigmoid(a)
    silu_a = a * tl.sigmoid(a)
    result = silu_a * b

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(a, b):
    """
    Fused SiLU(a) * b implemented as a single Triton kernel.
    Stages fused: sigmoid(a), a*sigmoid(a) [SiLU], multiply by b, cast to bf16, store.
    """
    assert a.shape == b.shape, "Shape mismatch"
    assert a.device == b.device, "Device mismatch"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"

    out = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _silu_mul_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out