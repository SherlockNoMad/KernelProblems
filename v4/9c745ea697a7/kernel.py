"""
Fused kernel implementation for add -> RMS norm -> add -> RMS norm backward -> add -> dtype conversion.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_norm_forward_kernel(
    # Input tensors
    mm_3_ptr, embedding_ptr, weight_ptr,
    # Output tensors
    out1_ptr, out2_ptr,
    # Intermediate storage for backward pass
    add_tensor_ptr, rstd_ptr,
    # Tensor dimensions
    seq_len, hidden_dim,
    # Strides
    mm_3_stride_0, mm_3_stride_1,
    embedding_stride_0, embedding_stride_1, embedding_stride_2,
    weight_stride_0,
    out_stride_0, out_stride_1,
    add_tensor_stride_0, add_tensor_stride_1, add_tensor_stride_2,
    # Constants
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for the current sequence position
    pid = tl.program_id(0)
    
    # Calculate row offset
    row_start = pid
    
    # First pass: compute addition and store intermediate result
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Load inputs - mm_3 needs reshaping from [8192, 4096] to [1, 8192, 4096]
        mm_3_offset = row_start * mm_3_stride_0 + offsets * mm_3_stride_1
        mm_3_vals = tl.load(mm_3_ptr + mm_3_offset, mask=mask, other=0.0)
        
        # Load embedding [1, 8192, 4096]
        embedding_offset = 0 * embedding_stride_0 + row_start * embedding_stride_1 + offsets * embedding_stride_2
        embedding_vals = tl.load(embedding_ptr + embedding_offset, mask=mask, other=0.0)
        
        # First addition: embedding + reshape(mm_3)
        add_tensor_vals = embedding_vals + mm_3_vals
        
        # Store intermediate result for backward pass
        add_tensor_offset = 0 * add_tensor_stride_0 + row_start * add_tensor_stride_1 + offsets * add_tensor_stride_2
        tl.store(add_tensor_ptr + add_tensor_offset, add_tensor_vals, mask=mask)
    
    # Second pass: compute RMS norm statistics for the entire row
    sum_sq = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Reload add_tensor values
        add_tensor_offset = 0 * add_tensor_stride_0 + row_start * add_tensor_stride_1 + offsets * add_tensor_stride_2
        add_tensor_vals = tl.load(add_tensor_ptr + add_tensor_offset, mask=mask, other=0.0)
        
        # Convert to float32 and accumulate sum of squares
        add_tensor_f32 = add_tensor_vals.to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, add_tensor_f32 * add_tensor_f32, 0.0))
    
    # Compute RMS statistics
    mean_sq = sum_sq / hidden_dim
    rstd_val = 1.0 / tl.sqrt(mean_sq + eps)
    
    # Store rstd for backward pass
    tl.store(rstd_ptr + row_start, rstd_val)
    
    # Third pass: apply normalization and store outputs
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Reload inputs
        add_tensor_offset = 0 * add_tensor_stride_0 + row_start * add_tensor_stride_1 + offsets * add_tensor_stride_2
        add_tensor_vals = tl.load(add_tensor_ptr + add_tensor_offset, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offsets * weight_stride_0, mask=mask, other=0.0)
        
        # Apply RMS normalization
        add_tensor_f32 = add_tensor_vals.to(tl.float32)
        weight_f32 = weight_vals.to(tl.float32)
        normalized = add_tensor_f32 * rstd_val * weight_f32
        
        # Store outputs (both out1 and out2 are the same reshaped tensor)
        out_offset = row_start * out_stride_0 + offsets * out_stride_1
        normalized_bf16 = normalized.to(tl.bfloat16)
        tl.store(out1_ptr + out_offset, normalized_bf16, mask=mask)
        tl.store(out2_ptr + out_offset, normalized_bf16, mask=mask)


@triton.jit
def _fused_rms_norm_backward_kernel(
    # Input tensors for backward pass
    mm_664_ptr, mm_666_ptr, add_tensor_ptr, weight_ptr, add_218_ptr,
    # Intermediate values from forward pass
    rstd_ptr,
    # Output tensors
    out3_ptr, out4_ptr,
    # Tensor dimensions
    seq_len, hidden_dim,
    # Strides
    mm_stride_0, mm_stride_1,
    add_tensor_stride_0, add_tensor_stride_1, add_tensor_stride_2,
    weight_stride_0,
    add_218_stride_0, add_218_stride_1, add_218_stride_2,
    out3_stride_0, out3_stride_1, out3_stride_2,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Load rstd for this row
    rstd_val = tl.load(rstd_ptr + pid)
    
    # First pass: compute correction terms for RMS norm backward
    sum_grad_weight = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Load grad outputs (mm_664 and mm_666 reshaped to [1, 8192, 4096])
        mm_664_offset = pid * mm_stride_0 + offsets * mm_stride_1
        mm_666_offset = pid * mm_stride_0 + offsets * mm_stride_1
        mm_664_vals = tl.load(mm_664_ptr + mm_664_offset, mask=mask, other=0.0)
        mm_666_vals = tl.load(mm_666_ptr + mm_666_offset, mask=mask, other=0.0)
        
        # Load normalized input and weight
        add_tensor_offset = 0 * add_tensor_stride_0 + pid * add_tensor_stride_1 + offsets * add_tensor_stride_2
        add_tensor_vals = tl.load(add_tensor_ptr + add_tensor_offset, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offsets * weight_stride_0, mask=mask, other=0.0)
        
        # Compute grad_output: mm_664 + mm_666
        grad_output = mm_664_vals + mm_666_vals
        
        # Convert to float32 for computation
        grad_output_f32 = grad_output.to(tl.float32)
        add_tensor_f32 = add_tensor_vals.to(tl.float32)
        weight_f32 = weight_vals.to(tl.float32)
        
        # Accumulate sum for correction term: sum(grad_output * weight * input)
        sum_grad_weight += tl.sum(tl.where(mask, grad_output_f32 * weight_f32 * add_tensor_f32, 0.0))
    
    # Compute correction factor for RMS norm backward
    correction_factor = sum_grad_weight * rstd_val * rstd_val * rstd_val / hidden_dim
    
    # Second pass: compute gradients and final output
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Reload all inputs
        mm_664_offset = pid * mm_stride_0 + offsets * mm_stride_1
        mm_666_offset = pid * mm_stride_0 + offsets * mm_stride_1
        mm_664_vals = tl.load(mm_664_ptr + mm_664_offset, mask=mask, other=0.0)
        mm_666_vals = tl.load(mm_666_ptr + mm_666_offset, mask=mask, other=0.0)
        
        add_tensor_offset = 0 * add_tensor_stride_0 + pid * add_tensor_stride_1 + offsets * add_tensor_stride_2
        add_tensor_vals = tl.load(add_tensor_ptr + add_tensor_offset, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offsets * weight_stride_0, mask=mask, other=0.0)
        add_218_offset = 0 * add_218_stride_0 + pid * add_218_stride_1 + offsets * add_218_stride_2
        add_218_vals = tl.load(add_218_ptr + add_218_offset, mask=mask, other=0.0)
        
        # Compute grad_output: mm_664 + mm_666
        grad_output = mm_664_vals + mm_666_vals
        
        # Convert to float32 for computation
        grad_output_f32 = grad_output.to(tl.float32)
        add_tensor_f32 = add_tensor_vals.to(tl.float32)
        weight_f32 = weight_vals.to(tl.float32)
        
        # RMS norm backward formula:
        # grad_input = rstd * weight * grad_output - correction_factor * input
        grad_input = rstd_val * weight_f32 * grad_output_f32 - correction_factor * add_tensor_f32
        
        # Final addition: add_218 + grad_input
        final_output = add_218_vals + grad_input.to(tl.bfloat16)
        
        # Store output 3
        out3_offset = 0 * out3_stride_0 + pid * out3_stride_1 + offsets * out3_stride_2
        tl.store(out3_ptr + out3_offset, final_output, mask=mask)


@triton.jit
def _weight_gradient_kernel(
    # Input tensors
    mm_664_ptr, mm_666_ptr, add_tensor_ptr, rstd_ptr,
    # Output tensor
    out4_ptr,
    # Tensor dimensions
    seq_len, hidden_dim,
    # Strides
    mm_stride_0, mm_stride_1,
    add_tensor_stride_0, add_tensor_stride_1, add_tensor_stride_2,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    # Get the feature dimension index
    feature_idx = tl.program_id(0)
    
    if feature_idx >= hidden_dim:
        return
    
    # Accumulate weight gradient across all sequence positions
    weight_grad_sum = 0.0
    
    for seq_idx in range(seq_len):
        # Load grad outputs
        mm_664_val = tl.load(mm_664_ptr + seq_idx * mm_stride_0 + feature_idx * mm_stride_1)
        mm_666_val = tl.load(mm_666_ptr + seq_idx * mm_stride_0 + feature_idx * mm_stride_1)
        
        # Load normalized input
        add_tensor_val = tl.load(add_tensor_ptr + 0 * add_tensor_stride_0 + seq_idx * add_tensor_stride_1 + feature_idx * add_tensor_stride_2)
        
        # Load rstd
        rstd_val = tl.load(rstd_ptr + seq_idx)
        
        # Compute grad_output
        grad_output = mm_664_val + mm_666_val
        
        # Accumulate weight gradient: sum over sequence of (grad_output * normalized_input)
        weight_grad_sum += grad_output.to(tl.float32) * add_tensor_val.to(tl.float32) * rstd_val
    
    # Store weight gradient
    tl.store(out4_ptr + feature_idx, weight_grad_sum)


def kernel_function(mm_3, embedding, getitem_787, mm_664, mm_666, add_218):
    """
    Fused kernel wrapper for the sequence:
    add -> _fused_rms_norm -> add -> _fused_rms_norm_backward -> add -> _to_copy
    
    This implementation fuses multiple operations to minimize memory traffic:
    1. Forward RMS normalization with input additions
    2. Backward RMS normalization with output additions and dtype conversion
    
    Cannot further fuse the weight gradient computation with the backward pass
    because it requires a reduction across the sequence dimension while the
    backward pass operates per-sequence-position.
    """
    
    # Validate inputs
    device = mm_3.device
    dtype = mm_3.dtype
    assert dtype == torch.bfloat16, f"Expected bfloat16, got {dtype}"
    
    # Get dimensions
    batch_size, seq_len, hidden_dim = embedding.shape
    assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
    
    # Ensure all tensors are contiguous and properly shaped
    mm_3 = mm_3.contiguous()
    embedding = embedding.contiguous()
    mm_664 = mm_664.contiguous()
    mm_666 = mm_666.contiguous()
    add_218 = add_218.contiguous()
    
    # Handle getitem_787 properly - use reshape instead of view
    weight = getitem_787.contiguous().reshape(hidden_dim)
    
    # Allocate outputs with correct shapes
    out1 = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)  # reshape_default_1
    out2 = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)  # reshape_default_2
    out3 = torch.empty([batch_size, seq_len, hidden_dim], dtype=torch.bfloat16, device=device)   # add_tensor_2
    out4 = torch.empty([hidden_dim], dtype=torch.float32, device=device)                         # _to_copy_default
    
    # Allocate intermediate storage
    add_tensor = torch.empty([batch_size, seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    rstd = torch.empty([seq_len], dtype=torch.float32, device=device)
    
    # Reshape inputs for processing - use reshape instead of view
    mm_664_reshaped = mm_664.reshape(batch_size, seq_len, hidden_dim)
    mm_666_reshaped = mm_666.reshape(batch_size, seq_len, hidden_dim)
    
    # Constants
    BLOCK_SIZE = 512
    
    # Launch forward pass kernel
    grid = (seq_len,)
    _fused_rms_norm_forward_kernel[grid](
        mm_3, embedding, weight,
        out1, out2,
        add_tensor, rstd,
        seq_len, hidden_dim,
        # Strides
        mm_3.stride(0), mm_3.stride(1),
        embedding.stride(0), embedding.stride(1), embedding.stride(2),
        weight.stride(0),
        out1.stride(0), out1.stride(1),
        add_tensor.stride(0), add_tensor.stride(1), add_tensor.stride(2),
        1e-05, BLOCK_SIZE
    )
    
    # Launch backward pass kernel
    _fused_rms_norm_backward_kernel[grid](
        mm_664_reshaped, mm_666_reshaped, add_tensor, weight, add_218,
        rstd,
        out3, out4,
        seq_len, hidden_dim,
        # Strides
        mm_664_reshaped.stride(1), mm_664_reshaped.stride(2),
        add_tensor.stride(0), add_tensor.stride(1), add_tensor.stride(2),
        weight.stride(0),
        add_218.stride(0), add_218.stride(1), add_218.stride(2),
        out3.stride(0), out3.stride(1), out3.stride(2),
        BLOCK_SIZE
    )
    
    # Launch weight gradient kernel for proper reduction
    grid_weight = (triton.cdiv(hidden_dim, 1),)
    _weight_gradient_kernel[grid_weight](
        mm_664_reshaped, mm_666_reshaped, add_tensor, rstd,
        out4,
        seq_len, hidden_dim,
        # Strides
        mm_664_reshaped.stride(1), mm_664_reshaped.stride(2),
        add_tensor.stride(0), add_tensor.stride(1), add_tensor.stride(2),
        1
    )
    
    return (out1, out2, out3, out4)