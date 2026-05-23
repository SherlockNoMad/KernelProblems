"""
Kernel implementation for fused _to_copy operation with transpose.
This kernel fuses:
1. Transpose operations (which cancel out)
2. Dtype conversion from bf16 to f32
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
def dtype_convert_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing:
    - Transpose operations (cancel out, so effectively identity)
    - Dtype conversion from bfloat16 to float32
    """
    # Calculate the starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load input data (bf16) with masking
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert to float32 (the transpose operations cancel out)
    output_data = input_data.to(tl.float32)
    
    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)

def kernel_function(mm_665):
    """
    Wrapper function that implements the fused region:
    - t.default(mm_665) -> transpose
    - t.default(transposed) -> transpose back (cancels out)  
    - _to_copy.default(..., dtype=torch.float32) -> dtype conversion
    
    The two transpose operations cancel out, so this is effectively
    just a dtype conversion from bfloat16 to float32.
    """
    # Validate input
    assert mm_665.dtype == torch.bfloat16, f"Expected bfloat16 input, got {mm_665.dtype}"
    assert mm_665.device.type == 'cuda', f"Expected CUDA tensor, got {mm_665.device}"
    
    # Calculate total number of elements
    n_elements = mm_665.numel()
    
    # Allocate output tensor with same shape but float32 dtype
    # Since transposes cancel out, output shape is same as input
    output = torch.empty_like(mm_665, dtype=torch.float32, device=mm_665.device)
    
    # Launch kernel with 1D grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    dtype_convert_kernel[grid](
        mm_665,
        output, 
        n_elements,
    )
    
    return output