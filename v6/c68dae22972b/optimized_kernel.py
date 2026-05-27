import triton
import triton.language as tl
import torch


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    af = a.to(tl.float32)
    silu_a = af * tl.sigmoid(af)
    result = silu_a * b.to(tl.float32)

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_221, mm_222):
    a = mm_221.contiguous()
    b = mm_222.contiguous()

    output = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_kernel[grid](a, b, output, n_elements,
                           BLOCK_SIZE=BLOCK_SIZE,
                           num_warps=8, num_stages=4)

    return output