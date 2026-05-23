"""
Kernel implementation for fused transpose and dtype conversion operations.
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
    Fused kernel for:
    1. Double transpose (identity operation)
    2. Dtype conversion from bf16 to f32
    
    Since t(t(x)) = x, we only need to perform dtype conversion.
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load input data (bf16)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32
    output_data = input_data.to(tl.float32)
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Wrapper function that performs fused transpose and dtype conversion.
    
    Args:
        input_tensor: Input tensor of shape [4096, 14336] with dtype bf16
        
    Returns:
        Output tensor of same shape with dtype float32
    """
    # Validate input
    assert input_tensor.dtype == torch.bfloat16, f"Expected bf16, got {input_tensor.dtype}"
    assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
    
    # Calculate total elements
    n_elements = input_tensor.numel()
    
    # Allocate output tensor with float32 dtype
    output_tensor = torch.empty_like(input_tensor, dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    dtype_conversion_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
    )
    
    return output_tensor