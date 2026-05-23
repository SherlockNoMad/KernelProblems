"""
Fused kernel for reshape and dtype conversion operations.
Performs: bf16[8192, 4096] -> f32[1, 8192, 32, 64, 2] (duplicated twice)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_reshape_convert_kernel(
    input_ptr,
    output1_ptr,
    output2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Load bfloat16 data
    2. Convert to float32
    3. Store to two identical output tensors
    
    The reshaping is handled by the memory layout - we just need to copy
    the data with dtype conversion since reshape operations don't change
    the underlying data order for contiguous tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load bfloat16 data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 (Triton handles this automatically)
    output_data = input_data.to(tl.float32)
    
    # Store to both output tensors
    tl.store(output1_ptr + offsets, output_data, mask=mask)
    tl.store(output2_ptr + offsets, output_data, mask=mask)


def kernel_function(mm_217):
    """
    Wrapper function that handles the fused reshape and dtype conversion.
    
    Input: bf16[8192, 4096]
    Output: Two identical f32[1, 8192, 32, 64, 2] tensors
    
    Fusion strategy: Since all operations are just reshapes and dtype conversion,
    we can fuse everything into a single memory copy with dtype conversion.
    The reshaping is handled by creating output tensors with the correct shape.
    """
    # Validate input
    assert mm_217.dtype == torch.bfloat16, f"Expected bfloat16, got {mm_217.dtype}"
    assert mm_217.shape == (8192, 4096), f"Expected shape [8192, 4096], got {mm_217.shape}"
    assert mm_217.device.type == "cuda", f"Expected CUDA tensor, got {mm_217.device}"
    
    # Calculate total elements
    n_elements = mm_217.numel()  # 8192 * 4096 = 33,554,432
    
    # Allocate output tensors with the final shape [1, 8192, 32, 64, 2]
    # Total elements should be the same: 1 * 8192 * 32 * 64 * 2 = 33,554,432
    output_shape = [1, 8192, 32, 64, 2]
    output1 = torch.empty(output_shape, dtype=torch.float32, device=mm_217.device)
    output2 = torch.empty(output_shape, dtype=torch.float32, device=mm_217.device)
    
    # Verify element count matches
    assert output1.numel() == n_elements, f"Element count mismatch: {output1.numel()} != {n_elements}"
    
    # Launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch fused kernel
    fused_reshape_convert_kernel[grid](
        mm_217,
        output1,
        output2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (output1, output2)