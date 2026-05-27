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

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    silu = af * tl.sigmoid(af)
    result = (silu * bf).to(tl.bfloat16)

    tl.store(out_ptr + offsets, result, mask=mask, cache_modifier=".cs")


def kernel_function(a, b):
    a_contig = a.contiguous()
    b_contig = b.contiguous()
    output = torch.empty_like(a_contig)
    n_elements = a_contig.numel()

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_kernel[grid](
        a_contig, b_contig, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return output