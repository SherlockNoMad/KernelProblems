"""
Fused RMS Norm + Add + RMS Norm Backward + Add + Copy kernel implementation.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_forward_kernel(
    # Inputs
    input_ptr, weight_ptr,
    # Outputs  
    output_ptr, inv_rms_ptr,
    # Dimensions
    seq_len, hidden_dim,
    # Strides
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    # Constants
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward RMS normalization kernel."""
    row_idx = tl.program_id(0)
    
    if row_idx >= seq_len:
        return
        
    # Load entire row for RMS computation
    row_start = row_idx * input_stride_0
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Load input row
    x = tl.load(input_ptr + row_start + offsets * input_stride_1, mask=mask, other=0.0).to(tl.float32)
    
    # Load weights
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    
    # Compute RMS normalization exactly as PyTorch does
    x_squared = x * x
    mean_sq = tl.sum(x_squared, axis=0) / hidden_dim
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    
    # Normalize and apply weight
    x_normed = x * inv_rms
    output = x_normed * weight
    
    # Store outputs
    output_start = row_idx * output_stride_0
    tl.store(output_ptr + output_start + offsets * output_stride_1, output.to(tl.bfloat16), mask=mask)
    
    # Store inverse RMS for backward pass
    if row_idx < seq_len:
        tl.store(inv_rms_ptr + row_idx, inv_rms)


@triton.jit  
def rms_norm_backward_kernel(
    # Inputs
    grad_output_ptr, input_ptr, weight_ptr, inv_rms_ptr,
    # Outputs
    grad_input_ptr, grad_weight_ptr,
    # Dimensions  
    seq_len, hidden_dim,
    # Strides
    grad_output_stride_0, grad_output_stride_1,
    input_stride_0, input_stride_1,
    grad_input_stride_0, grad_input_stride_1,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    """Backward RMS normalization kernel."""
    row_idx = tl.program_id(0)
    
    if row_idx >= seq_len:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Load inputs
    grad_out = tl.load(grad_output_ptr + row_idx * grad_output_stride_0 + offsets * grad_output_stride_1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(input_ptr + row_idx * input_stride_0 + offsets * input_stride_1, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    inv_rms = tl.load(inv_rms_ptr + row_idx).to(tl.float32)
    
    # Compute normalized input
    x_normed = x * inv_rms
    
    # RMS norm backward pass - following PyTorch's implementation
    # grad_input = (grad_out * weight - (x_normed * sum(grad_out * weight * x_normed)) / N) * inv_rms
    grad_weight_local = grad_out * x_normed
    grad_normed = grad_out * weight
    
    # Compute the correction term
    sum_term = tl.sum(grad_normed * x_normed, axis=0) / hidden_dim
    grad_input = (grad_normed - x_normed * sum_term) * inv_rms
    
    # Store grad_input
    tl.store(grad_input_ptr + row_idx * grad_input_stride_0 + offsets * grad_input_stride_1, grad_input.to(tl.bfloat16), mask=mask)
    
    # Store grad_weight contribution for this row
    tl.store(grad_weight_ptr + row_idx * hidden_dim + offsets, grad_weight_local, mask=mask)


@triton.jit
def reduce_grad_weight_kernel(
    grad_weight_contributions_ptr, grad_weight_ptr,
    seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce grad_weight contributions across sequence length."""
    col_idx = tl.program_id(0)
    
    if col_idx >= hidden_dim:
        return
        
    # Sum contributions across all sequence positions
    total = 0.0
    for i in range(seq_len):
        contrib = tl.load(grad_weight_contributions_ptr + i * hidden_dim + col_idx).to(tl.float32)
        total += contrib
    
    # Store final grad_weight
    tl.store(grad_weight_ptr + col_idx, total)


def kernel_function(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420):
    """
    Fused kernel for RMS norm operations.
    
    This implementation performs the following operations:
    1. _fused_rms_norm: Forward RMS normalization
    2. add: Addition of mm_230 and mm_232  
    3. _fused_rms_norm_backward: Backward pass of RMS norm
    4. add: Addition with getitem_420
    5. _to_copy: Type conversion to float32
    
    Technical fusion limitation: The RMS norm backward pass requires a reduction across
    the hidden dimension for each sequence position, followed by a separate reduction
    across sequence positions for the weight gradients. These different reduction
    patterns and the need for intermediate storage of inv_rms values make complete
    fusion into a single kernel technically challenging without significant memory
    overhead and synchronization complexity.
    """
    
    # Input validation and shape extraction
    device = add_62_recomputed.device
    batch_size, seq_len, hidden_dim = add_62_recomputed.shape
    
    # Process weight tensor following the exact reference sequence
    # view.dtype -> clone -> _unsafe_view
    weight_viewed = getitem_1624.view(dtype=torch.bfloat16)
    weight_cloned = weight_viewed.clone(memory_format=torch.contiguous_format)
    weight_processed = weight_cloned.view([hidden_dim])
    
    # Allocate intermediate and output tensors
    normed_output = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    inv_rms = torch.empty([seq_len], dtype=torch.float32, device=device)
    
    # Forward RMS norm
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    if BLOCK_SIZE > 4096:  # Limit to avoid resource issues
        BLOCK_SIZE = 4096
    
    # Flatten the input for processing
    input_flattened = add_62_recomputed.view(-1, hidden_dim)
    
    rms_norm_forward_kernel[(seq_len,)](
        input_flattened, weight_processed,
        normed_output, inv_rms,
        seq_len, hidden_dim,
        input_flattened.stride(0), input_flattened.stride(1),
        normed_output.stride(0), normed_output.stride(1),
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape outputs for first two return values (same data, different views)
    reshape_default = normed_output
    reshape_default_1 = normed_output
    
    # Add mm_230 and mm_232 to create grad_output
    # Reshape mm tensors to match expected dimensions
    mm_230_reshaped = mm_230.view(1, seq_len, hidden_dim)  
    mm_232_reshaped = mm_232.view(1, seq_len, hidden_dim)
    
    # Perform addition (this is a simple elementwise operation)
    grad_output = mm_230_reshaped + mm_232_reshaped
    
    # Allocate for backward pass
    grad_input = torch.empty([1, seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    grad_weight_contributions = torch.empty([seq_len, hidden_dim], dtype=torch.float32, device=device)
    
    # Flatten tensors for backward kernel
    grad_output_flat = grad_output.view(-1, hidden_dim)
    grad_input_flat = grad_input.view(-1, hidden_dim)
    
    # Backward RMS norm  
    rms_norm_backward_kernel[(seq_len,)](
        grad_output_flat, input_flattened, 
        weight_processed, inv_rms,
        grad_input_flat, grad_weight_contributions,
        seq_len, hidden_dim,
        grad_output_flat.stride(0), grad_output_flat.stride(1),
        input_flattened.stride(0), input_flattened.stride(1),
        grad_input_flat.stride(0), grad_input_flat.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final addition with getitem_420
    add_tensor_1 = getitem_420 + grad_input
    
    # Reduce grad_weight contributions
    grad_weight_final = torch.empty([hidden_dim], dtype=torch.float32, device=device)
    
    reduce_grad_weight_kernel[(hidden_dim,)](
        grad_weight_contributions, grad_weight_final,
        seq_len, hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (reshape_default, reshape_default_1, add_tensor_1, grad_weight_final)