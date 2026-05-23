"""
Fused SiLU -> Multiply kernel implementation using Triton.
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
    key=['n_elements'],
)
@triton.jit
def fused_silu_mul_kernel(
    input1_ptr,  # mm_221
    input2_ptr,  # mm_222  
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing: output = silu(input1) * input2
    
    Fused stages:
    1. Load input1 and input2 elements
    2. Compute SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    3. Element-wise multiply with input2
    4. Store result
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary conditions
    mask = offsets < n_elements
    
    # Load input tensors with masking
    input1 = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    input2 = tl.load(input2_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for computation precision
    input1_f32 = input1.to(tl.float32)
    input2_f32 = input2.to(tl.float32)
    
    # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    # Using numerically stable implementation
    neg_input1 = -input1_f32
    exp_neg_input1 = tl.exp(neg_input1)
    sigmoid_input1 = 1.0 / (1.0 + exp_neg_input1)
    silu_result = input1_f32 * sigmoid_input1
    
    # Element-wise multiply with input2
    result_f32 = silu_result * input2_f32
    
    # Convert back to original dtype and store
    result = result_f32.to(input1_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, result, mask=mask)


def kernel_function(mm_221, mm_222):
    """
    Wrapper function for fused SiLU -> multiply operation.
    
    Performs: output = silu(mm_221) * mm_222
    
    Args:
        mm_221: Input tensor for SiLU activation [8192, 14336], bfloat16
        mm_222: Input tensor for multiplication [8192, 14336], bfloat16
        
    Returns:
        Output tensor with same shape and dtype as inputs
    """
    # Input validation
    assert mm_221.shape == mm_222.shape, f"Shape mismatch: {mm_221.shape} vs {mm_222.shape}"
    assert mm_221.dtype == mm_222.dtype, f"Dtype mismatch: {mm_221.dtype} vs {mm_222.dtype}"
    assert mm_221.device == mm_222.device, f"Device mismatch: {mm_221.device} vs {mm_222.device}"
    
    # Calculate total number of elements
    n_elements = mm_221.numel()
    
    # Allocate output tensor
    output = torch.empty_like(mm_221)
    
    # Launch kernel with 1D grid
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_silu_mul_kernel[grid](
        mm_221,
        mm_222, 
        output,
        n_elements,
    )
    
    return output