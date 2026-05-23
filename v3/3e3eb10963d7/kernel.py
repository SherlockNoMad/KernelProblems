"""
Fused kernel for reshape + dtype conversion + transpose operations.
Performs: f32[1,8192,8,64,2] -> reshape -> bf16 conversion -> reshape -> transpose -> bf16[1024,8192]
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def fused_reshape_dtype_transpose_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing:
    1. Read from input f32[1,8192,8,64,2] (flattened)
    2. Convert to bf16
    3. Write to transposed output location bf16[1024,8192]
    
    The logical transformation is:
    - Input: [1,8192,8,64,2] -> flatten to [8192*8*64*2] = [8192*1024]
    - After reshapes: [8192,1024] 
    - After transpose: [1024,8192]
    """
    
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load input data (f32)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to bfloat16
    output_data = input_data.to(tl.bfloat16)
    
    # Calculate transposed indices
    # Original logical shape after reshapes: [8192, 1024]
    # We want to transpose to: [1024, 8192]
    # For each linear index i, find (row, col) in [8192, 1024]
    # Then write to (col, row) position in [1024, 8192]
    
    row_indices = offsets // 1024  # Which row in [8192, 1024]
    col_indices = offsets % 1024   # Which col in [8192, 1024]
    
    # Transposed position: (col, row) in [1024, 8192]
    # Linear index in transposed tensor: col * 8192 + row
    transposed_offsets = col_indices * 8192 + row_indices
    
    # Store to transposed locations
    tl.store(output_ptr + transposed_offsets, output_data, mask=mask)


def kernel_function(view_as_real_126):
    """
    Wrapper function for fused reshape + dtype conversion + transpose.
    
    Args:
        view_as_real_126: Input tensor f32[1,8192,8,64,2]
        
    Returns:
        Output tensor bf16[1024,8192] (transposed)
    """
    # Validate input
    assert view_as_real_126.dtype == torch.float32, f"Expected float32, got {view_as_real_126.dtype}"
    assert view_as_real_126.shape == (1, 8192, 8, 64, 2), f"Expected shape (1,8192,8,64,2), got {view_as_real_126.shape}"
    assert view_as_real_126.is_cuda, "Input must be on CUDA device"
    
    # Calculate dimensions
    # After logical reshapes: [8192, 1024] -> transpose -> [1024, 8192]
    numel = 8192 * 1024  # Total elements to process
    
    # Allocate output tensor with transposed shape
    output = torch.empty((1024, 8192), dtype=torch.bfloat16, device=view_as_real_126.device)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(numel, META['BLOCK_SIZE']),)
    
    fused_reshape_dtype_transpose_kernel[grid](
        view_as_real_126,  # Input pointer (Triton handles flattening)
        output,            # Output pointer
        numel,
    )
    
    return output