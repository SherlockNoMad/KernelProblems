"""
Fused kernel implementing transpose -> transpose -> dtype conversion.
Since double transpose is identity, this fuses into pure dtype conversion.
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
def dtype_conversion_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel stages:
    1. Load bf16 input
    2. Convert to f32 
    3. Store f32 output
    
    Note: Double transpose (t -> t) is mathematically identity, so we fuse
    the entire operation into elementwise dtype conversion for optimal performance.
    """
    # Calculate the current block's starting position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load bf16 input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert bf16 to f32 (this handles the dtype conversion stage)
    output_data = input_data.to(tl.float32)
    
    # Store f32 output data
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Wrapper function implementing fused transpose->transpose->dtype_conversion.
    
    Args:
        input_tensor: bf16[128256, 4096] input tensor
        
    Returns:
        f32[128256, 4096] output tensor
    """
    # Validate input
    assert input_tensor.dtype == torch.bfloat16, f"Expected bf16 input, got {input_tensor.dtype}"
    assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
    assert input_tensor.is_contiguous(), "Input tensor must be contiguous"
    
    # Calculate total elements
    n_elements = input_tensor.numel()
    
    # Allocate output tensor with f32 dtype, same shape as input
    output_tensor = torch.empty_like(input_tensor, dtype=torch.float32, device=input_tensor.device)
    
    # Launch configuration
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the fused kernel
    dtype_conversion_kernel[grid](
        input_tensor,
        output_tensor, 
        n_elements,
    )
    
    return output_tensor