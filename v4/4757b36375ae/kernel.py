"""
Fused reshape + add kernel implementation.
This kernel performs:
1. Reshape mm_216 from [8192, 4096] to [1, 8192, 4096] 
2. Element-wise addition with add_60 [1, 8192, 4096]
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
def fused_reshape_add_kernel(
    mm_ptr,      # Input matrix to reshape [8192, 4096]
    add_ptr,     # Input tensor to add [1, 8192, 4096]
    output_ptr,  # Output tensor [1, 8192, 4096]
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for this block
    pid = tl.program_id(axis=0)
    
    # Calculate the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load data from both input tensors
    # mm_216 is already in the right memory layout for [1, 8192, 4096]
    # since reshape doesn't change memory layout, just view
    mm_data = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
    add_data = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    result = mm_data + add_data
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)


def kernel_function(mm_216, add_60):
    """
    Fused reshape + add operation.
    
    Args:
        mm_216: Input tensor [8192, 4096] to be reshaped to [1, 8192, 4096]
        add_60: Input tensor [1, 8192, 4096] to add
        
    Returns:
        Output tensor [1, 8192, 4096] = reshaped mm_216 + add_60
    """
    # Validate inputs
    assert mm_216.shape == (8192, 4096), f"Expected mm_216 shape [8192, 4096], got {mm_216.shape}"
    assert add_60.shape == (1, 8192, 4096), f"Expected add_60 shape [1, 8192, 4096], got {add_60.shape}"
    assert mm_216.dtype == add_60.dtype, f"Dtype mismatch: {mm_216.dtype} vs {add_60.dtype}"
    assert mm_216.device == add_60.device, f"Device mismatch: {mm_216.device} vs {add_60.device}"
    
    # Calculate total elements
    n_elements = mm_216.numel()  # 8192 * 4096 = 33,554,432 elements
    
    # Allocate output tensor with target shape [1, 8192, 4096]
    output = torch.empty((1, 8192, 4096), dtype=mm_216.dtype, device=mm_216.device)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    fused_reshape_add_kernel[grid](
        mm_216,      # Input tensor (will be treated as reshaped)
        add_60,      # Tensor to add
        output,      # Output tensor
        n_elements,  # Total elements to process
    )
    
    return output