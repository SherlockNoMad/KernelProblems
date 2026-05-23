import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=4, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def dtype_conversion_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output_data = input_data.to(tl.float32)
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    assert input_tensor.dtype == torch.bfloat16
    assert input_tensor.is_cuda

    input_contiguous = input_tensor.contiguous()
    n_elements = input_contiguous.numel()
    output_tensor = torch.empty(input_contiguous.shape, dtype=torch.float32, device=input_contiguous.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    dtype_conversion_kernel[grid](
        input_contiguous,
        output_tensor,
        n_elements,
    )

    return output_tensor