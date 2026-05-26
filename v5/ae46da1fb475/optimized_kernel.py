import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(mm_ptr, add_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(add_ptr + offsets, mask=mask, other=0.0)

    result = a + b

    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(mm_220, add_61):
    out_shape = (1, 8192, 4096)
    out0 = torch.empty(out_shape, dtype=torch.bfloat16, device=mm_220.device)

    mm_flat = mm_220.view(-1)
    add_flat = add_61.view(-1)
    out_flat = out0.view(-1)

    n_elements = mm_flat.numel()
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_kernel[grid](
        mm_flat, add_flat, out_flat, n_elements,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4,
    )

    return out0, out0