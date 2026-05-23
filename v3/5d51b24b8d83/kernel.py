"""
Fused kernel for reshape and dtype conversion operations on attention key projections.
This kernel fuses: reshape -> reshape -> _to_copy (dtype conversion) -> reshape
Produces two identical outputs from the same input processing.
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def reshape_dtype_kernel(
    input_ptr,
    output1_ptr,
    output2_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Load bfloat16 data
    2. Convert to float32 (_to_copy operation)
    3. Store to two identical output tensors with reshaped layout
    
    The logical reshaping is handled by stride calculations in the wrapper.
    This kernel focuses on the dtype conversion which is the compute-intensive part.
    """
    # Calculate program position
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary conditions
    mask = offsets < numel
    
    # Load input data (bfloat16)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 (this is the _to_copy operation)
    output_data = input_data.to(tl.float32)
    
    # Store to both output tensors (they are identical)
    tl.store(output1_ptr + offsets, output_data, mask=mask)
    tl.store(output2_ptr + offsets, output_data, mask=mask)

def kernel_function(mm_218):
    """
    Wrapper function that performs fused reshape and dtype conversion.
    
    Fuses the operations:
    - Path 1: reshape -> reshape -> _to_copy -> reshape  
    - Path 2: reshape -> reshape -> _to_copy -> reshape
    
    Both paths are identical, so we compute once and produce two outputs.
    """
    # Validate input
    assert mm_218.device.type == 'cuda', f"Expected CUDA tensor, got {mm_218.device}"
    assert mm_218.dtype == torch.bfloat16, f"Expected bfloat16, got {mm_218.dtype}"
    assert list(mm_218.shape) == [8192, 1024], f"Expected shape [8192, 1024], got {mm_218.shape}"
    
    # Calculate total number of elements
    numel = mm_218.numel()  # 8192 * 1024 = 8,388,608
    
    # Allocate output tensors with final target shape [1, 8192, 8, 64, 2]
    # Total elements remain the same, just reshaped
    target_shape = [1, 8192, 8, 64, 2]
    output1 = torch.empty(target_shape, dtype=torch.float32, device=mm_218.device)
    output2 = torch.empty(target_shape, dtype=torch.float32, device=mm_218.device)
    
    # Flatten tensors for kernel processing (kernel works on flattened data)
    input_flat = mm_218.view(-1)
    output1_flat = output1.view(-1) 
    output2_flat = output2.view(-1)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    
    reshape_dtype_kernel[grid](
        input_flat,
        output1_flat, 
        output2_flat,
        numel,
    )
    
    # Return tensors with correct final shapes
    # The kernel handles the dtype conversion, PyTorch handles the logical reshaping
    return (output1, output2)