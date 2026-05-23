"""
Fused embedding lookup kernel implementation.
This kernel fuses view operations with embedding lookup to minimize memory traffic.
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
    key=['seq_len', 'embed_dim'],
)
@triton.jit
def fused_embedding_kernel(
    # Input tensors
    input_ptr,          # wait_tensor_871: source data for embedding table
    indices_ptr,        # arg583_1: embedding indices 
    output_ptr,         # output tensor
    # Tensor dimensions
    batch_size,         # batch dimension
    seq_len,           # sequence length 
    embed_dim,         # embedding dimension
    vocab_size,        # vocabulary size
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Get program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    
    # Calculate dimension offsets for this block
    dim_start = dim_block_idx * BLOCK_SIZE
    dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_offsets < embed_dim
    
    # Load the embedding index for this batch and sequence position
    indices_offset = batch_idx * seq_len + seq_idx
    embedding_idx = tl.load(indices_ptr + indices_offset)
    
    # Calculate input offsets for the embedding table
    # The embedding table starts from the input tensor after view/split operations
    embedding_row_start = embedding_idx * embed_dim
    input_offsets = embedding_row_start + dim_offsets
    input_mask = (embedding_idx < vocab_size) & dim_mask
    
    # Load embedding values
    embedding_values = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Calculate output offset
    output_offset = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + dim_offsets
    
    # Store results
    tl.store(output_ptr + output_offset, embedding_values, mask=dim_mask)


def kernel_function(wait_tensor_871, arg583_1):
    """
    Fused embedding lookup implementation.
    
    This kernel fuses the following operations:
    1. View operations to reshape input tensor
    2. Split operation to extract embedding table portion  
    3. Embedding lookup using indices
    
    Args:
        wait_tensor_871: Input tensor [525336576] containing embedding table data
        arg583_1: Indices tensor [1, 8192] for embedding lookup
        
    Returns:
        Output tensor [1, 8192, 4096] with embedded values
    """
    
    # Validate inputs
    assert wait_tensor_871.dtype == torch.bfloat16, f"Expected bfloat16, got {wait_tensor_871.dtype}"
    assert arg583_1.dtype == torch.int64, f"Expected int64, got {arg583_1.dtype}"
    assert wait_tensor_871.device == arg583_1.device, "Input tensors must be on same device"
    
    # Extract dimensions from the reference implementation
    batch_size, seq_len = arg583_1.shape  # [1, 8192]
    embed_dim = 4096  # From view_default_1 shape [128256, 4096]
    vocab_size = 128256  # From view_default_1 shape [128256, 4096]
    
    # Validate tensor sizes match expected dimensions
    expected_input_size = vocab_size * embed_dim  # 128256 * 4096 = 525336576
    assert wait_tensor_871.numel() >= expected_input_size, \
        f"Input tensor too small: {wait_tensor_871.numel()} < {expected_input_size}"
    
    # Validate indices are in valid range
    assert torch.all(arg583_1 >= 0) and torch.all(arg583_1 < vocab_size), \
        f"Indices out of range [0, {vocab_size})"
    
    # Allocate output tensor
    output = torch.empty(
        (batch_size, seq_len, embed_dim), 
        dtype=torch.bfloat16, 
        device=wait_tensor_871.device
    )
    
    # Calculate grid dimensions
    # We need to process each (batch, sequence, embedding_dim) combination
    BLOCK_SIZE = 128  # Will be selected by autotune
    dim_blocks = triton.cdiv(embed_dim, BLOCK_SIZE)
    
    def grid(META):
        return (batch_size, seq_len, triton.cdiv(embed_dim, META['BLOCK_SIZE']))
    
    # Launch kernel
    fused_embedding_kernel[grid](
        wait_tensor_871,
        arg583_1, 
        output,
        batch_size,
        seq_len,
        embed_dim,
        vocab_size,
    )
    
    return output