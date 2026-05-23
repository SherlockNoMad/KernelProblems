import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large block sizes to maximize memory throughput and reduce launch overhead
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_reshape_add_kernel(
    mm_ptr,
    add_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    mm_data = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
    add_data = tl.load(add_ptr + offsets, mask=mask, other=0.0)

    result = mm_data + add_data

    tl.store(output_ptr + offsets, result, mask=mask)


def kernel_function(mm_216, add_60):
    assert mm_216.shape == (8192, 4096)
    assert add_60.shape == (1, 8192, 4096)

    n_elements = mm_216.numel()

    # Ensure contiguous for coalesced access
    mm_216 = mm_216.contiguous()
    add_60_flat = add_60.contiguous()

    output = torch.empty((1, 8192, 4096), dtype=mm_216.dtype, device=mm_216.device)

    def grid(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    fused_reshape_add_kernel[grid](
        mm_216,
        add_60_flat,
        output,
        n_elements,
    )

    return output