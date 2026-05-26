import triton
import triton.language as tl
import torch


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    af = a.to(tl.float32)
    silu_a = af * tl.sigmoid(af)
    result = silu_a * b.to(tl.float32)

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask, cache_modifier=".cs")


def kernel_function(a, b):
    assert a.shape == b.shape
    assert a.is_cuda and b.is_cuda

    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out = torch.empty_like(a)
    out_flat = out.view(-1)
    n_elements = a_flat.numel()

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_kernel[grid](
        a_flat, b_flat, out_flat, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out