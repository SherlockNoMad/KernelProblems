import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    a_f = a.to(tl.float32)
    b_f = b.to(tl.float32)
    sig = tl.sigmoid(a_f)
    result = a_f * sig * b_f

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_4, mm_5):
    assert mm_4.is_cuda and mm_5.is_cuda
    assert mm_4.shape == mm_5.shape
    assert mm_4.dtype == mm_5.dtype

    a = mm_4.contiguous()
    b = mm_5.contiguous()
    out = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _silu_mul_kernel[grid](
        a, b, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return out