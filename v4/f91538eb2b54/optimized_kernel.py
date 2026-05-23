import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=4),
    ],
    key=['numel'],
)
@triton.jit
def fused_reshape_add_kernel(
    mm_220_ptr,
    add_61_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    mm_220_data = tl.load(mm_220_ptr + offsets, mask=mask, other=0.0)
    add_61_data = tl.load(add_61_ptr + offsets, mask=mask, other=0.0)

    result = mm_220_data + add_61_data

    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(mm_220, add_61):
    assert mm_220.shape == (8192, 4096)
    assert add_61.shape == (1, 8192, 4096)

    numel = mm_220.numel()
    output_shape = (1, 8192, 4096)
    out = torch.empty(output_shape, dtype=mm_220.dtype, device=mm_220.device)

    def grid(META):
        return (triton.cdiv(numel, META['BLOCK_SIZE']),)

    fused_reshape_add_kernel[grid](
        mm_220,
        add_61,
        out,
        numel,
    )

    return (out, out)