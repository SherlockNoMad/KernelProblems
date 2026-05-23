"""
Fused rotary positional embedding kernel implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    # Input tensors
    input_ptr, rotary_real_ptr, rotary_imag_ptr, output_ptr,
    # Dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for input tensor
    input_stride_batch, input_stride_head, input_stride_seq, input_stride_dim,
    # Strides for rotary tensor  
    rotary_stride_batch, rotary_stride_seq, rotary_stride_head, rotary_stride_dim,
    # Strides for output tensor
    output_stride_batch, output_stride_seq, output_stride_head, output_stride_dim,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1) 
    pid_head = tl.program_id(2)
    
    # Calculate base offsets
    input_base = (pid_batch * input_stride_batch + 
                  pid_head * input_stride_head + 
                  pid_seq * input_stride_seq)
    
    rotary_base = (pid_batch * rotary_stride_batch +
                   pid_seq * rotary_stride_seq + 
                   pid_head * rotary_stride_head)
    
    output_base = (pid_batch * output_stride_batch +
                   pid_seq * output_stride_seq +
                   pid_head * output_stride_head)
    
    # Process head_dim elements in blocks
    for block_start in range(0, head_dim, BLOCK_SIZE):
        # Calculate dimension offsets
        dim_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        dim_mask = dim_offsets < head_dim
        
        # Load input data (convert bfloat16 to float32 for computation)
        input_ptrs = input_ptr + input_base + dim_offsets * input_stride_dim
        input_data = tl.load(input_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
        
        # For complex operations, we need pairs of elements (real, imag)
        # dim_offsets represents the flattened complex view indices
        complex_idx = dim_offsets // 2  # Which complex number
        is_real = (dim_offsets % 2) == 0  # Real or imaginary part
        
        # Load rotary embedding components
        rotary_ptrs = rotary_real_ptr + rotary_base + complex_idx * rotary_stride_dim
        rotary_real = tl.load(rotary_ptrs, mask=(complex_idx < (head_dim // 2)) & dim_mask, other=1.0)
        
        rotary_ptrs_imag = rotary_imag_ptr + rotary_base + complex_idx * rotary_stride_dim  
        rotary_imag = tl.load(rotary_ptrs_imag, mask=(complex_idx < (head_dim // 2)) & dim_mask, other=0.0)
        
        # Get the paired element (real<->imag)
        paired_offsets = tl.where(is_real, dim_offsets + 1, dim_offsets - 1)
        paired_mask = (paired_offsets >= 0) & (paired_offsets < head_dim) & dim_mask
        paired_ptrs = input_ptr + input_base + paired_offsets * input_stride_dim
        paired_data = tl.load(paired_ptrs, mask=paired_mask, other=0.0).to(tl.float32)
        
        # Extract real and imaginary parts of input
        input_real = tl.where(is_real, input_data, paired_data)
        input_imag = tl.where(is_real, paired_data, input_data)
        
        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        result_real = input_real * rotary_real - input_imag * rotary_imag
        result_imag = input_real * rotary_imag + input_imag * rotary_real
        
        # Select the appropriate component for this position
        result = tl.where(is_real, result_real, result_imag)
        
        # Store result (convert back to bfloat16)
        output_ptrs = output_ptr + output_base + dim_offsets * output_stride_dim
        tl.store(output_ptrs, result.to(tl.bfloat16), mask=dim_mask)


@triton.jit  
def simple_transpose_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, num_heads, head_dim,
    input_stride_batch, input_stride_head, input_stride_seq, input_stride_dim,
    output_stride_batch, output_stride_seq, output_stride_head, output_stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    # Calculate base offsets
    input_base = (pid_batch * input_stride_batch + 
                  pid_head * input_stride_head + 
                  pid_seq * input_stride_seq)
    
    output_base = (pid_batch * output_stride_batch +
                   pid_seq * output_stride_seq +
                   pid_head * output_stride_head)
    
    # Process head_dim elements in blocks
    for block_start in range(0, head_dim, BLOCK_SIZE):
        dim_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        dim_mask = dim_offsets < head_dim
        
        # Load and store
        input_ptrs = input_ptr + input_base + dim_offsets * input_stride_dim
        input_data = tl.load(input_ptrs, mask=dim_mask, other=0.0)
        
        output_ptrs = output_ptr + output_base + dim_offsets * output_stride_dim  
        tl.store(output_ptrs, input_data, mask=dim_mask)


def kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643):
    """
    Fused rotary positional embedding kernel.
    
    Applies rotary embeddings to queries and keys, then transposes and reshapes.
    Operations fused:
    1. Q: transpose -> dtype_convert -> reshape -> complex_view -> rotary_mul -> real_view -> reshape -> dtype_convert -> final_transpose
    2. K: same as Q but with different rotary embedding
    3. V: simple transpose and reshape (no rotary embedding)
    """
    device = _conj_62.device
    
    # Input validation
    assert _conj_62.device == device and _conj_63.device == device
    assert getitem_641.device == device and getitem_642.device == device and getitem_643.device == device
    
    # Extract real and imaginary parts of complex tensors since Triton doesn't support complex64 directly
    # _conj_62: [1, 8192, 1, 64] complex64 -> split to real/imag float32
    conj_62_view = torch.view_as_real(_conj_62)  # [1, 8192, 1, 64, 2]
    conj_62_real = conj_62_view[..., 0].contiguous()  # [1, 8192, 1, 64]
    conj_62_imag = conj_62_view[..., 1].contiguous()  # [1, 8192, 1, 64]
    
    # _conj_63: [1, 8192, 1, 64] complex64 -> split to real/imag float32
    conj_63_view = torch.view_as_real(_conj_63)  # [1, 8192, 1, 64, 2]
    conj_63_real = conj_63_view[..., 0].contiguous()  # [1, 8192, 1, 64]
    conj_63_imag = conj_63_view[..., 1].contiguous()  # [1, 8192, 1, 64]
    
    # Extract dimensions
    batch_size = 1
    seq_len = 8192
    
    # Process queries with rotary embedding
    # getitem_641: [1, 32, 8192, 128] -> transpose -> [1, 8192, 32, 128]
    q_output = torch.empty([seq_len, 32 * 128], dtype=torch.bfloat16, device=device)
    
    # Launch rotary embedding kernel for queries
    def grid_q(meta):
        return (batch_size, seq_len, 32)
    
    rotary_embedding_kernel[grid_q](
        getitem_641, conj_63_real, conj_63_imag, q_output,
        batch_size, seq_len, 32, 128,
        # Input strides: [1, 32, 8192, 128] with strides [33554432, 128, 4096, 1]
        33554432, 128, 4096, 1,
        # Rotary strides: [1, 8192, 1, 64] -> broadcast to [1, 8192, 32, 64]
        8192 * 64, 64, 0, 1,  # head dimension broadcasts (stride 0)
        # Output strides: [8192, 4096] -> [seq_len, num_heads * head_dim]
        4096, 4096, 128, 1,
        BLOCK_SIZE=64,
    )
    
    # Transpose final query output
    q_final = q_output.t()  # [4096, 8192]
    
    # Process keys with rotary embedding  
    # getitem_642: [1, 8, 8192, 128] -> transpose -> [1, 8192, 8, 128]
    k_output = torch.empty([seq_len, 8 * 128], dtype=torch.bfloat16, device=device)
    
    def grid_k(meta):
        return (batch_size, seq_len, 8)
        
    rotary_embedding_kernel[grid_k](
        getitem_642, conj_62_real, conj_62_imag, k_output,
        batch_size, seq_len, 8, 128,
        # Input strides: [1, 8, 8192, 128] with strides [8388608, 128, 1024, 1]  
        8388608, 128, 1024, 1,
        # Rotary strides: [1, 8192, 1, 64] -> broadcast to [1, 8192, 8, 64]
        8192 * 64, 64, 0, 1,  # head dimension broadcasts
        # Output strides: [8192, 1024] -> [seq_len, num_heads * head_dim]
        1024, 1024, 128, 1,
        BLOCK_SIZE=64,
    )
    
    # Transpose final key output
    k_final = k_output.t()  # [1024, 8192]
    
    # Process values (no rotary embedding, just transpose and reshape)
    # getitem_643: [1, 8, 8192, 128] -> transpose -> [1, 8192, 8, 128] -> reshape -> [8192, 1024] -> transpose -> [1024, 8192]
    v_output = torch.empty([seq_len, 8 * 128], dtype=torch.bfloat16, device=device)
    
    def grid_v(meta):
        return (batch_size, seq_len, 8)
        
    simple_transpose_kernel[grid_v](
        getitem_643, v_output,
        batch_size, seq_len, 8, 128,
        # Input strides: [1, 8, 8192, 128] with strides [8388608, 128, 1024, 1]
        8388608, 128, 1024, 1,
        # Output strides: [8192, 1024]
        1024, 1024, 128, 1,
        BLOCK_SIZE=64,
    )
    
    # Transpose final value output
    v_final = v_output.t()  # [1024, 8192]
    
    return (q_final, k_final, v_final)