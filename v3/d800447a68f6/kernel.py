"""
Kernel implementation 4.
Fused reshape + dtype conversion + transpose kernel.
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
    ],
    key=['total_elements'],
)
@triton.jit
def fused_reshape_convert_transpose_kernel(
    input_ptr,
    output_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel stages:
    1. Load f32 data from input tensor
    2. Convert f32 → bf16  
    3. Store to transposed output position
    
    Input: f32[1, 8192, 32, 64, 2] → reshaped to [8192, 4096]
    Output: bf16[4096, 8192] (transposed)
    """
    # Get thread block starting position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data (f32)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert f32 → bf16
    bf16_data = input_data.to(tl.bfloat16)
    
    # Compute transposed output indices
    # Input logical shape after reshape: [8192, 4096] 
    # Output shape: [4096, 8192]
    # For transpose: output[j, i] = input[i, j]
    
    row = offsets // 4096  # i coordinate (0 to 8191)
    col = offsets % 4096   # j coordinate (0 to 4095)
    
    # Transposed output position: [col, row] in [4096, 8192] tensor
    output_offsets = col * 8192 + row
    output_mask = mask  # Same validity since we're just rearranging
    
    # Store to transposed position
    tl.store(output_ptr + output_offsets, bf16_data, mask=output_mask)


def kernel_function(view_as_real_127):
    """
    Fused implementation of:
    - reshape f32[1, 8192, 32, 64, 2] → [8192, 4096] 
    - convert f32 → bf16
    - transpose [8192, 4096] → [4096, 8192]
    """
    # Validate input
    assert view_as_real_127.dtype == torch.float32, f"Expected f32 input, got {view_as_real_127.dtype}"
    assert view_as_real_127.shape == (1, 8192, 32, 64, 2), f"Expected shape [1, 8192, 32, 64, 2], got {view_as_real_127.shape}"
    
    # Calculate dimensions
    # After first reshape: [1, 8192, 32, 128] → [8192, 4096]
    # After transpose: [4096, 8192]
    total_elements = 8192 * 4096  # 33,554,432 elements
    
    # Allocate output tensor
    output = torch.empty((4096, 8192), dtype=torch.bfloat16, device=view_as_real_127.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    fused_reshape_convert_transpose_kernel[grid](
        view_as_real_127,  # Input tensor (will be accessed linearly)
        output,            # Output tensor 
        total_elements,
    )
    
    return output