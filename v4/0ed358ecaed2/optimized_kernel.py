import torch
import triton
import triton.language as tl


@triton.jit
def dtype_convert_kernel_vec4(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = input_data.to(tl.float32)
    tl.store(output_ptr + offsets, output_data, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def dtype_convert_kernel_tuned(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = input_data.to(tl.float32)
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(mm_665):
    assert mm_665.dtype == torch.bfloat16
    assert mm_665.device.type == 'cuda'
    
    n_elements = mm_665.numel()
    
    # Ensure contiguous for vectorized access
    mm_665_contig = mm_665.contiguous()
    
    output = torch.empty(n_elements, dtype=torch.float32, device=mm_665.device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    dtype_convert_kernel_tuned[grid](
        mm_665_contig,
        output,
        n_elements,
    )
    
    return output.view(mm_665.shape)