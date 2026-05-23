"""
Fused RMS normalization kernel with tensor processing.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def fused_rms_norm_kernel(
    input_ptr, weight_ptr, output_ptr, rstd_ptr,
    N, hidden_size, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused RMS normalization kernel.
    Stages fused:
    1. RMS norm computation with weight scaling
    2. Store normalized output for subsequent reshaping
    """
    row_idx = tl.program_id(0)
    
    # Calculate row offsets
    input_row_start = row_idx * hidden_size
    output_row_start = row_idx * hidden_size
    
    # Process in blocks across the hidden dimension
    block_start = 0
    variance_sum = tl.zeros((), dtype=tl.float32)
    
    # First pass: compute variance
    for block_start in tl.range(0, hidden_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        
        input_ptrs = input_ptr + input_row_start + offsets
        vals = tl.load(input_ptrs, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        
        # Accumulate sum of squares for variance
        variance_sum += tl.sum(vals_f32 * vals_f32)
    
    # Compute RMS normalization factor
    mean_square = variance_sum / hidden_size
    rstd = 1.0 / tl.sqrt(mean_square + eps)
    
    # Store reciprocal standard deviation
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Second pass: normalize and apply weight
    for block_start in tl.range(0, hidden_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        
        input_ptrs = input_ptr + input_row_start + offsets
        output_ptrs = output_ptr + output_row_start + offsets
        weight_ptrs = weight_ptr + offsets
        
        vals = tl.load(input_ptrs, mask=mask, other=0.0)
        weights = tl.load(weight_ptrs, mask=mask, other=1.0)
        
        vals_f32 = vals.to(tl.float32)
        weights_f32 = weights.to(tl.float32)
        
        # Apply RMS normalization and weight scaling
        normalized = vals_f32 * rstd * weights_f32
        
        tl.store(output_ptrs, normalized.to(input_ptr.dtype.element_ty), mask=mask)

@triton.jit  
def tensor_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Tensor transpose kernel.
    Fused stage: transpose operation for weight matrices
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    input_ptrs = input_ptr + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1
    vals = tl.load(input_ptrs, mask=mask, other=0.0)
    
    output_ptrs = output_ptr + offs_n[:, None] * output_stride_0 + offs_m[None, :] * output_stride_1
    tl.store(output_ptrs, vals.T, mask=mask.T)

def kernel_function(getitem_1619, add_62, getitem_1620, getitem_1621, getitem_1622):
    """
    Fused RMS normalization with tensor processing wrapper.
    
    Fused operations:
    1. Weight tensor processing from getitem_1619
    2. RMS normalization of add_62 with processed weight
    3. Tensor reshaping and transposition of weight matrices
    """
    device = add_62.device
    dtype = add_62.dtype
    
    # Process weight tensor (from getitem_1619)  
    # Equivalent to: view_dtype -> clone -> unsafe_view operations
    weight_view = getitem_1619.view(dtype)
    weight_contiguous = weight_view.clone(memory_format=torch.contiguous_format)
    weight = weight_contiguous.view(4096)
    
    # Setup for RMS normalization
    batch_size, seq_len, hidden_size = add_62.shape
    N = batch_size * seq_len
    
    # Allocate outputs
    normalized_output = torch.empty_like(add_62)
    rstd = torch.empty((N,), device=device, dtype=torch.float32)
    
    # Launch RMS normalization kernel
    input_reshaped = add_62.view(N, hidden_size)
    output_reshaped = normalized_output.view(N, hidden_size)
    
    BLOCK_SIZE = 256
    grid = (N,)
    
    fused_rms_norm_kernel[grid](
        input_reshaped, weight, output_reshaped, rstd,
        N, hidden_size, 1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape outputs (equivalent to reshape operations in reference)
    reshape_default = normalized_output.reshape(8192, 4096)
    reshape_default_1 = normalized_output.reshape(8192, 4096)
    
    # Process weight matrices with view operations and transpose
    # Process getitem_1620 -> w1 weights
    weight1_view = getitem_1620.view(dtype)
    weight1_contiguous = weight1_view.clone(memory_format=torch.contiguous_format)
    weight1_reshaped = weight1_contiguous.view(14336, 4096)
    t_default = weight1_reshaped.t()
    
    # Process getitem_1621 -> w3 weights  
    weight2_view = getitem_1621.view(dtype)
    weight2_contiguous = weight2_view.clone(memory_format=torch.contiguous_format)
    weight2_reshaped = weight2_contiguous.view(14336, 4096)
    t_default_1 = weight2_reshaped.t()
    
    # Process getitem_1622 -> w2 weights
    weight3_view = getitem_1622.view(dtype)
    weight3_contiguous = weight3_view.clone(memory_format=torch.contiguous_format)
    weight3_reshaped = weight3_contiguous.view(4096, 14336)
    t_default_2 = weight3_reshaped.t()
    
    return (reshape_default, reshape_default_1, t_default, t_default_1, t_default_2)