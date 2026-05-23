"""
Fused kernel for reshape + dtype conversion + transpose operations.
Performs: reshape [1,8192,8,64,2] -> [1,8192,8,128], f32->bf16, transpose dims 1,2
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
def fused_reshape_convert_transpose_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing:
    1. Reshape [1, 8192, 8, 64, 2] -> [1, 8192, 8, 128]
    2. Convert f32 -> bf16
    3. Transpose dims 1,2: [1, 8192, 8, 128] -> [1, 8, 8192, 128]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # For each output element, compute corresponding input position
    # Output shape: [1, 8, 8192, 128]
    # Input shape after conceptual reshape: [1, 8192, 8, 128]
    
    # Decompose output linear index into multi-dimensional indices
    # output_idx = b * (8 * 8192 * 128) + h * (8192 * 128) + s * 128 + d
    # where b=0 (batch), h=head, s=seq, d=dim
    
    temp_idx = offsets
    d = temp_idx % 128
    temp_idx = temp_idx // 128
    s = temp_idx % 8192
    temp_idx = temp_idx // 8192
    h = temp_idx % 8
    b = temp_idx // 8
    
    # Map to input indices (before transpose, after reshape)
    # Input after reshape: [1, 8192, 8, 128]
    # So transpose swaps dims 1,2: (b, s, h, d) in input corresponds to (b, h, s, d) in output
    
    # Input linear index = b * (8192 * 8 * 128) + s * (8 * 128) + h * 128 + d
    input_linear_idx = b * (8192 * 8 * 128) + s * (8 * 128) + h * 128 + d
    
    # Load input data (f32) and convert to bf16
    input_data = tl.load(input_ptr + input_linear_idx, mask=mask, other=0.0)
    output_data = input_data.to(tl.bfloat16)
    
    # Store to output
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    """
    Fused implementation of reshape + dtype conversion + transpose.
    
    Args:
        input_tensor: Input tensor of shape [1, 8192, 8, 64, 2] with dtype float32
        
    Returns:
        Output tensor of shape [1, 8, 8192, 128] with dtype bfloat16
    """
    # Validate input
    assert input_tensor.shape == (1, 8192, 8, 64, 2), f"Expected shape [1, 8192, 8, 64, 2], got {input_tensor.shape}"
    assert input_tensor.dtype == torch.float32, f"Expected dtype float32, got {input_tensor.dtype}"
    assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
    
    # Output shape after all operations: [1, 8, 8192, 128]
    output_shape = (1, 8, 8192, 128)
    output_tensor = torch.empty(output_shape, dtype=torch.bfloat16, device=input_tensor.device)
    
    # Total number of elements in output
    numel = output_tensor.numel()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    
    fused_reshape_convert_transpose_kernel[grid](
        input_tensor,
        output_tensor,
        numel,
    )
    
    return output_tensor