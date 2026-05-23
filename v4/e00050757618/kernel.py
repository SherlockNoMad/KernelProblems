"""
Kernel implementation 4.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_to_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel implementing:
    1. Double transpose (no-op for shape preservation)  
    2. Dtype conversion from bf16 to f32
    
    Since double transpose returns to original layout, this fuses into
    a single elementwise dtype conversion pass.
    """
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary conditions
    mask = offsets < n_elements
    
    # Load input data (bf16)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 (the double transpose is a no-op)
    output_data = input_data.to(tl.float32)
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Wrapper function that handles the fused _to_copy operation.
    
    Implements: double transpose + dtype conversion (bf16 -> f32)
    Since double transpose is identity, this fuses to elementwise conversion.
    """
    # Validate input
    assert input_tensor.dtype == torch.bfloat16, f"Expected bf16 input, got {input_tensor.dtype}"
    assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
    
    # Calculate total elements
    n_elements = input_tensor.numel()
    
    # Allocate output tensor (same shape, f32 dtype)
    output = torch.empty_like(input_tensor, dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_to_copy_kernel[grid](
        input_tensor,
        output,
        n_elements,
    )
    
    return output