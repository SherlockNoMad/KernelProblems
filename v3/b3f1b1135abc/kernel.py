"""
Fused attention tensor processing kernel implementation.
Performs transpose, dtype conversion, and reshape operations for query, key, and value tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def query_processing_kernel(
    input_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    input_stride_b, input_stride_h, input_stride_s, input_stride_d,
    output_stride_b, output_stride_s, output_stride_h, output_stride_d, output_stride_split,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for query processing:
    - Transpose from [B, H, S, D] to [B, S, H, D]
    - Convert from bfloat16 to float32
    - Reshape to [B, S, H, D//2, 2]
    """
    # Get program ID and calculate indices
    pid = tl.program_id(axis=0)
    
    # Calculate total elements and block boundaries
    total_elements = batch_size * seq_len * num_heads * head_dim
    start_idx = pid * BLOCK_SIZE
    
    # Create offset array for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate multi-dimensional indices from linear offset
    # We're processing in output order: [batch, seq, head, dim]
    batch_idx = offsets // (seq_len * num_heads * head_dim)
    remainder = offsets % (seq_len * num_heads * head_dim)
    seq_idx = remainder // (num_heads * head_dim)
    remainder = remainder % (num_heads * head_dim)
    head_idx = remainder // head_dim
    dim_idx = remainder % head_dim
    
    # Calculate input addresses (original layout: [B, H, S, D])
    input_offsets = (batch_idx * input_stride_b + 
                    head_idx * input_stride_h + 
                    seq_idx * input_stride_s + 
                    dim_idx * input_stride_d)
    
    # Load and convert to float32
    input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    output_data = input_data.to(tl.float32)
    
    # Calculate output addresses for reshaped tensor [B, S, H, D//2, 2]
    split_dim = dim_idx // 2  # D//2 dimension
    split_idx = dim_idx % 2   # 2 dimension
    
    output_offsets = (batch_idx * output_stride_b + 
                     seq_idx * output_stride_s + 
                     head_idx * output_stride_h + 
                     split_dim * output_stride_d + 
                     split_idx * output_stride_split)
    
    # Store result
    tl.store(output_ptr + output_offsets, output_data, mask=mask)


@triton.jit
def key_processing_kernel(
    input_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    input_stride_b, input_stride_h, input_stride_s, input_stride_d,
    output_stride_b, output_stride_s, output_stride_h, output_stride_d, output_stride_split,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for key processing:
    - Transpose from [B, H, S, D] to [B, S, H, D]  
    - Convert from bfloat16 to float32
    - Reshape to [B, S, H, D//2, 2]
    """
    # Get program ID and calculate indices
    pid = tl.program_id(axis=0)
    
    # Calculate total elements and block boundaries
    total_elements = batch_size * seq_len * num_heads * head_dim
    start_idx = pid * BLOCK_SIZE
    
    # Create offset array for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate multi-dimensional indices from linear offset
    batch_idx = offsets // (seq_len * num_heads * head_dim)
    remainder = offsets % (seq_len * num_heads * head_dim)
    seq_idx = remainder // (num_heads * head_dim)
    remainder = remainder % (num_heads * head_dim)
    head_idx = remainder // head_dim
    dim_idx = remainder % head_dim
    
    # Calculate input addresses (original layout: [B, H, S, D])
    input_offsets = (batch_idx * input_stride_b + 
                    head_idx * input_stride_h + 
                    seq_idx * input_stride_s + 
                    dim_idx * input_stride_d)
    
    # Load and convert to float32
    input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    output_data = input_data.to(tl.float32)
    
    # Calculate output addresses for reshaped tensor [B, S, H, D//2, 2]
    split_dim = dim_idx // 2  # D//2 dimension  
    split_idx = dim_idx % 2   # 2 dimension
    
    output_offsets = (batch_idx * output_stride_b + 
                     seq_idx * output_stride_s + 
                     head_idx * output_stride_h + 
                     split_dim * output_stride_d + 
                     split_idx * output_stride_split)
    
    # Store result
    tl.store(output_ptr + output_offsets, output_data, mask=mask)


@triton.jit
def value_processing_kernel(
    input_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    input_stride_b, input_stride_h, input_stride_s, input_stride_d,
    output_stride_0, output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for value processing:
    - Transpose from [B, H, S, D] to [B, S, H, D]
    - Reshape to [S, H*D] 
    - Transpose to [H*D, S]
    Final output: [1024, 8192] (since H*D = 8*128 = 1024)
    """
    # Get program ID and calculate indices
    pid = tl.program_id(axis=0)
    
    # Calculate total elements and block boundaries
    total_elements = seq_len * num_heads * head_dim  # Final output size
    start_idx = pid * BLOCK_SIZE
    
    # Create offset array for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices in final output space [H*D, S]
    hd_idx = offsets // seq_len  # H*D dimension
    s_idx = offsets % seq_len    # S dimension
    
    # Map back to original input indices [B, H, S, D]
    # Since batch_size=1, we can ignore batch dimension
    h_idx = hd_idx // head_dim   # Head index
    d_idx = hd_idx % head_dim    # Dimension index
    
    # Calculate input addresses (original layout: [B, H, S, D])
    input_offsets = (h_idx * input_stride_h + 
                    s_idx * input_stride_s + 
                    d_idx * input_stride_d)
    
    # Load input data (keep as bfloat16)
    input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Calculate output addresses for transposed tensor [H*D, S]
    output_offsets = hd_idx * output_stride_0 + s_idx * output_stride_1
    
    # Store result
    tl.store(output_ptr + output_offsets, input_data, mask=mask)


def kernel_function(getitem_641, getitem_642, getitem_643):
    """
    Wrapper function that processes attention tensors with fused operations.
    
    Args:
        getitem_641: Query tensor [1, 32, 8192, 128] bfloat16
        getitem_642: Key tensor [1, 8, 8192, 128] bfloat16  
        getitem_643: Value tensor [1, 8, 8192, 128] bfloat16
    
    Returns:
        Tuple of processed tensors:
        - Query: [1, 8192, 32, 64, 2] float32
        - Key: [1, 8192, 8, 64, 2] float32
        - Value: [1024, 8192] bfloat16
    """
    device = getitem_641.device
    
    # Process Query tensor
    # Target shape: [1, 8192, 32, 64, 2]
    query_output = torch.empty((1, 8192, 32, 64, 2), dtype=torch.float32, device=device)
    
    query_total_elements = 1 * 8192 * 32 * 128
    BLOCK_SIZE = 256
    query_grid = (triton.cdiv(query_total_elements, BLOCK_SIZE),)
    
    query_processing_kernel[query_grid](
        getitem_641, query_output,
        1, 32, 8192, 128,  # batch_size, num_heads, seq_len, head_dim
        getitem_641.stride(0), getitem_641.stride(1), getitem_641.stride(2), getitem_641.stride(3),
        query_output.stride(0), query_output.stride(1), query_output.stride(2), query_output.stride(3), query_output.stride(4),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Process Key tensor  
    # Target shape: [1, 8192, 8, 64, 2]
    key_output = torch.empty((1, 8192, 8, 64, 2), dtype=torch.float32, device=device)
    
    key_total_elements = 1 * 8192 * 8 * 128
    key_grid = (triton.cdiv(key_total_elements, BLOCK_SIZE),)
    
    key_processing_kernel[key_grid](
        getitem_642, key_output,
        1, 8, 8192, 128,  # batch_size, num_heads, seq_len, head_dim
        getitem_642.stride(0), getitem_642.stride(1), getitem_642.stride(2), getitem_642.stride(3),
        key_output.stride(0), key_output.stride(1), key_output.stride(2), key_output.stride(3), key_output.stride(4),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Process Value tensor
    # Target shape: [1024, 8192] (transposed from [8192, 1024])
    value_output = torch.empty((1024, 8192), dtype=torch.bfloat16, device=device)
    
    value_total_elements = 8192 * 1024
    value_grid = (triton.cdiv(value_total_elements, BLOCK_SIZE),)
    
    value_processing_kernel[value_grid](
        getitem_643, value_output,
        1, 8, 8192, 128,  # batch_size, num_heads, seq_len, head_dim
        getitem_643.stride(0), getitem_643.stride(1), getitem_643.stride(2), getitem_643.stride(3),
        value_output.stride(0), value_output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (query_output, key_output, value_output)