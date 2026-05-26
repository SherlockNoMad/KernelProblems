import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(mm_ptr, add_ptr, out0_ptr, out1_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(mm_ptr + offsets, mask=mask)
    b = tl.load(add_ptr + offsets, mask=mask)

    result = a + b

    tl.store(out0_ptr + offsets, result, mask=mask)
    tl.store(out1_ptr + offsets, result, mask=mask)


def kernel_function(mm_220, add_61):
    """
    Fused kernel computing two identical outputs:
        out0 = add_61 + mm_220.view(1, 8192, 4096)
        out1 = add_61 + mm_220.view(1, 8192, 4096)

    Fusion: Since both outputs are identical, we compute the sum once
    and store it into two output buffers in a single kernel pass.
    """
    assert mm_220.is_cuda and add_61.is_cuda
    assert mm_220.dtype == torch.bfloat16 and add_61.dtype == torch.bfloat16
    assert mm_220.numel() == add_61.numel()

    out_shape = (1, 8192, 4096)
    out0 = torch.empty(out_shape, dtype=torch.bfloat16, device=mm_220.device)
    out1 = torch.empty(out_shape, dtype=torch.bfloat16, device=mm_220.device)

    # Ensure contiguous for flat indexing
    mm_flat = mm_220.contiguous().view(-1)
    add_flat = add_61.contiguous().view(-1)

    n_elements = mm_flat.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_kernel[grid](mm_flat, add_flat, out0, out1, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out0, out1