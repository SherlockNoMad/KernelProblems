"""
Fused kernel for reshape -> dtype conversion (f32->bf16) -> transpose operation.
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_reshape_dtype_transpose_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    # Input dimensions after conceptual reshape [1, 8192, 32, 128]
    dim1, dim2, dim3,  # 8192, 32, 128
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data (f32)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to bfloat16
    output_data = input_data.to(tl.bfloat16)
    
    # Calculate output indices for transpose
    # Input logical shape after reshape: [1, 8192, 32, 128] 
    # We need to transpose dims 1,2: [1, 8192, 32, 128] -> [1, 32, 8192, 128]
    
    # For each linear input index, compute the corresponding output index
    # Input index decomposition: idx = i1*32*128 + i2*128 + i3 (ignoring batch dim)
    # Output index: out_idx = i2*8192*128 + i1*128 + i3
    
    # Decompose linear indices into 3D coordinates (ignoring batch dimension)
    linear_idx = offsets
    i3 = linear_idx % dim3  # 128
    temp = linear_idx // dim3
    i2 = temp % dim2  # 32  
    i1 = temp // dim2  # 8192
    
    # Compute transposed output indices
    # Transpose: [dim1, dim2, dim3] -> [dim2, dim1, dim3]
    output_indices = i2 * (dim1 * dim3) + i1 * dim3 + i3
    
    # Store to output with transposed indexing
    tl.store(output_ptr + output_indices, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Fused kernel wrapper for reshape -> dtype conversion -> transpose.
    
    Input: [1, 8192, 32, 64, 2] f32
    Operations:
    1. Reshape to [1, 8192, 32, 128] 
    2. Convert f32 -> bf16
    3. Transpose to [1, 32, 8192, 128]
    
    Returns: [1, 32, 8192, 128] bf16
    """
    # Validate input
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if input_tensor.dtype != torch.float32:
        raise TypeError("Input must be float32")
        
    if list(input_tensor.shape) != [1, 8192, 32, 64, 2]:
        raise ValueError(f"Expected shape [1, 8192, 32, 64, 2], got {list(input_tensor.shape)}")
    
    # Calculate dimensions
    # After reshape: [1, 8192, 32, 128]
    # After transpose: [1, 32, 8192, 128]  
    batch_size = 1
    dim1, dim2, dim3 = 8192, 32, 128
    total_elements = batch_size * dim1 * dim2 * dim3
    
    # Allocate output tensor with final shape and dtype
    output_shape = [1, 32, 8192, 128]  # After transpose
    output = torch.empty(output_shape, dtype=torch.bfloat16, device=input_tensor.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    fused_reshape_dtype_transpose_kernel[grid](
        input_tensor,
        output,
        total_elements,
        dim1, dim2, dim3,
    )
    
    return output