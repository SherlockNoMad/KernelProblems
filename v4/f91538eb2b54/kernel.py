"""
Fused kernel for reshape + add operations.
Performs: reshape(mm_220, [1, 8192, 4096]) + add_61 (twice)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def fused_reshape_add_kernel(
    mm_220_ptr,
    add_61_ptr, 
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing:
    1. Reshape mm_220 from [8192, 4096] to [1, 8192, 4096] (no-op, just reinterpret)
    2. Element-wise add with add_61 [1, 8192, 4096]
    
    Both inputs have the same total elements, so we can process them linearly.
    """
    # Get program ID and compute element indices
    pid = tl.program_id(axis=0)
    
    # Compute the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary conditions
    mask = offsets < numel
    
    # Load data from both tensors
    # mm_220 is [8192, 4096] but we treat it as flattened
    # add_61 is [1, 8192, 4096] but we treat it as flattened
    mm_220_data = tl.load(mm_220_ptr + offsets, mask=mask, other=0.0)
    add_61_data = tl.load(add_61_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    result = mm_220_data + add_61_data
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(mm_220, add_61):
    """
    Wrapper function that handles the fused reshape + add operations.
    
    Args:
        mm_220: Input tensor [8192, 4096] 
        add_61: Input tensor [1, 8192, 4096]
        
    Returns:
        Tuple of (result1, result2) where both are identical [1, 8192, 4096] tensors
    """
    # Validate inputs
    assert mm_220.shape == (8192, 4096), f"Expected mm_220 shape [8192, 4096], got {mm_220.shape}"
    assert add_61.shape == (1, 8192, 4096), f"Expected add_61 shape [1, 8192, 4096], got {add_61.shape}"
    assert mm_220.dtype == add_61.dtype, f"Dtype mismatch: {mm_220.dtype} vs {add_61.dtype}"
    assert mm_220.device == add_61.device, f"Device mismatch: {mm_220.device} vs {add_61.device}"
    
    # Calculate total number of elements
    numel = mm_220.numel()  # 8192 * 4096 = 33,554,432
    assert numel == add_61.numel(), "Element count mismatch"
    
    # Allocate output tensor with target shape [1, 8192, 4096]
    output_shape = (1, 8192, 4096)
    out = torch.empty(output_shape, dtype=mm_220.dtype, device=mm_220.device)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(numel, META['BLOCK_SIZE']),)
    
    fused_reshape_add_kernel[grid](
        mm_220,
        add_61, 
        out,
        numel,
    )
    
    # Return tuple of two identical results as required by the test
    # Both operations in the reference model are identical, so we return the same result twice
    return (out, out)