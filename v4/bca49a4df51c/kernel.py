"""
Fused kernel implementing transpose + transpose + dtype conversion (bf16 to f32).
Since double transpose is identity operation, this effectively performs dtype conversion.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_transpose_dtype_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing:
    1. Double transpose (identity operation for shape)
    2. Dtype conversion from bf16 to f32
    
    Since double transpose doesn't change the tensor layout,
    this is effectively an elementwise dtype conversion.
    """
    # Calculate the starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data (bf16) with masking
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 (the double transpose is identity, so no layout change needed)
    output_data = input_data.to(tl.float32)
    
    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Wrapper function that performs fused transpose + transpose + dtype conversion.
    
    Args:
        input_tensor: Input tensor of shape [1024, 4096] with dtype bf16
        
    Returns:
        Output tensor of same shape with dtype f32
    """
    # Validate input
    assert input_tensor.dtype == torch.bfloat16, f"Expected bf16 input, got {input_tensor.dtype}"
    assert input_tensor.device.type == 'cuda', f"Expected CUDA tensor, got {input_tensor.device}"
    
    # Get tensor properties
    shape = input_tensor.shape
    n_elements = input_tensor.numel()
    
    # Allocate output tensor with f32 dtype
    output_tensor = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Launch kernel with 1D grid
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the fused kernel
    fused_transpose_dtype_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
    )
    
    return output_tensor