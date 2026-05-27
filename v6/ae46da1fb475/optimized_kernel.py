import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_kernel(
    mm_ptr, add_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
    add = tl.load(add_ptr + offsets, mask=mask, other=0.0)

    result = add + mm

    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(mm_220, add_61):
    assert mm_220.is_cuda and add_61.is_cuda
    assert mm_220.numel() == add_61.numel()

    out = torch.empty_like(add_61)

    n_elements = add_61.numel()
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_add_kernel[grid](
        mm_220, add_61, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    return (out, out)