"""
Fused RMS normalization with forward/backward passes and tensor operations.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def fused_rms_norm_forward_kernel(
    # Inputs
    add_1_ptr, weight_ptr,
    # Outputs
    norm_out_ptr, inv_rms_ptr,
    # Parameters
    batch_size, seq_len, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward RMS normalization kernel."""
    row_idx = tl.program_id(0)
    
    if row_idx >= batch_size * seq_len:
        return
    
    # Calculate row offset
    row_start = row_idx * hidden_size
    
    # Load input row in chunks if needed
    num_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    
    # Accumulate sum of squares across all blocks
    sum_squares = 0.0
    
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        
        x = tl.load(add_1_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.sum(x * x)
    
    # Compute RMS normalization factor
    mean_square = sum_squares / hidden_size
    rms = tl.sqrt(mean_square + eps)
    inv_rms = 1.0 / rms
    
    # Store inverse RMS for backward pass
    tl.store(inv_rms_ptr + row_idx, inv_rms)
    
    # Apply normalization and weight scaling in chunks
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        
        # Load input and weight
        x = tl.load(add_1_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        
        # Apply RMS normalization and weight scaling
        normalized = x * inv_rms
        output = normalized * weight
        
        # Store result
        tl.store(norm_out_ptr + row_start + col_offsets, output.to(tl.bfloat16), mask=mask)

@triton.jit
def fused_rms_norm_backward_kernel(
    # Inputs
    grad_output_ptr, input_ptr, inv_rms_ptr, weight_ptr,
    # Outputs
    grad_input_ptr, grad_weight_ptr,
    # Parameters
    batch_size, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward RMS normalization kernel."""
    row_idx = tl.program_id(0)
    
    if row_idx >= batch_size * seq_len:
        return
        
    row_start = row_idx * hidden_size
    inv_rms = tl.load(inv_rms_ptr + row_idx).to(tl.float32)
    
    num_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    
    # First pass: compute sum for gradient calculation
    sum_grad_x_norm_x = 0.0
    
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        
        # Load inputs
        grad_out = tl.load(grad_output_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        
        # Gradient w.r.t. normalized input
        grad_x_norm = grad_out * weight
        
        # Accumulate for RMS backward
        sum_grad_x_norm_x += tl.sum(grad_x_norm * x)
    
    # Second pass: compute gradients
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        
        # Load inputs
        grad_out = tl.load(grad_output_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        
        # Normalized input
        x_norm = x * inv_rms
        
        # Gradient w.r.t. weight (accumulate across batch)
        grad_weight_local = grad_out * x_norm
        
        # Gradient w.r.t. normalized input
        grad_x_norm = grad_out * weight
        
        # Gradient w.r.t. input (RMS norm backward pass)
        grad_input = inv_rms * grad_x_norm - (inv_rms * inv_rms * inv_rms * sum_grad_x_norm_x * x) / hidden_size
        
        # Store results
        tl.store(grad_input_ptr + row_start + col_offsets, grad_input.to(tl.bfloat16), mask=mask)
        
        # Store weight gradients (need proper reduction across batch dimension)
        tl.atomic_add(grad_weight_ptr + col_offsets, grad_weight_local, mask=mask)

@triton.jit  
def tensor_add_kernel(
    # Inputs
    a_ptr, b_ptr, c_ptr,
    # Output
    out_ptr,
    # Parameters
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for adding three tensors element-wise."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < numel
    
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + offset, mask=mask, other=0.0) 
    c = tl.load(c_ptr + offset, mask=mask, other=0.0)
    
    result = a + b + c
    tl.store(out_ptr + offset, result, mask=mask)

@triton.jit
def tensor_add_2_kernel(
    # Inputs
    a_ptr, b_ptr,
    # Output
    out_ptr,
    # Parameters
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for adding two tensors element-wise."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < numel
    
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + offset, mask=mask, other=0.0)
    
    result = a + b
    tl.store(out_ptr + offset, result, mask=mask)

def kernel_function(getitem_818, add_1, mm_656, mm_658, mm_660, add_215):
    """
    Wrapper function implementing the complete fused operation sequence.
    
    Fusion stages:
    1. Forward RMS normalization (add_1 input, getitem_818 weight)
    2. Tensor addition (mm_656 + mm_658 + mm_660) 
    3. Backward RMS normalization
    4. Final addition and type conversion
    
    Note: Complete fusion into single kernel is not feasible due to data dependencies
    between forward and backward passes requiring intermediate storage of inv_rms values
    and different memory access patterns for the various tensor shapes.
    """
    
    device = add_1.device
    batch_size, seq_len, hidden_size = add_1.shape
    
    # Validate input shapes
    assert getitem_818.numel() >= hidden_size, "Weight tensor too small"
    assert mm_656.shape == (seq_len, hidden_size), "mm_656 shape mismatch"
    assert mm_658.shape == (seq_len, hidden_size), "mm_658 shape mismatch"  
    assert mm_660.shape == (seq_len, hidden_size), "mm_660 shape mismatch"
    assert add_215.shape == (batch_size, seq_len, hidden_size), "add_215 shape mismatch"
    
    # Extract weight from getitem_818 (it's a strided view)
    weight = getitem_818.contiguous()[:hidden_size]
    
    # Allocate intermediate and output tensors
    norm_output = torch.empty((batch_size * seq_len, hidden_size), dtype=torch.bfloat16, device=device)
    inv_rms = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=device)
    
    # Stage 1: Forward RMS normalization
    BLOCK_SIZE = min(1024, triton.next_power_of_2(min(hidden_size, 1024)))
    
    grid = (batch_size * seq_len,)
    fused_rms_norm_forward_kernel[grid](
        add_1.view(-1), weight,
        norm_output, inv_rms,
        batch_size, seq_len, hidden_size,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 if BLOCK_SIZE >= 512 else 4,
    )
    
    # Stage 2: Add three matrix multiply results
    add_result = torch.empty_like(mm_656, dtype=torch.bfloat16, device=device)
    
    BLOCK_SIZE_ADD = 1024
    grid_add = (triton.cdiv(mm_656.numel(), BLOCK_SIZE_ADD),)
    tensor_add_kernel[grid_add](
        mm_656, mm_658, mm_660,
        add_result,
        mm_656.numel(),
        BLOCK_SIZE=BLOCK_SIZE_ADD,
        num_warps=4,
    )
    
    # Reshape add_result to match expected backward input shape
    add_result_reshaped = add_result.view(1, seq_len, hidden_size).expand(batch_size, -1, -1).contiguous()
    
    # Stage 3: Backward RMS normalization
    grad_input = torch.empty_like(add_1, dtype=torch.bfloat16, device=device)
    grad_weight = torch.zeros((hidden_size,), dtype=torch.float32, device=device)
    
    fused_rms_norm_backward_kernel[grid](
        add_result_reshaped.view(-1), add_1.view(-1), inv_rms, weight,
        grad_input.view(-1), grad_weight,
        batch_size, seq_len, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 if BLOCK_SIZE >= 512 else 4,
    )
    
    # Stage 4: Final addition  
    final_result = torch.empty_like(add_215, dtype=torch.bfloat16, device=device)
    
    grid_final = (triton.cdiv(add_215.numel(), BLOCK_SIZE_ADD),)
    tensor_add_2_kernel[grid_final](
        add_215, grad_input,
        final_result,
        add_215.numel(),
        BLOCK_SIZE=BLOCK_SIZE_ADD,
        num_warps=4,
    )
    
    # Prepare outputs with correct shapes
    # Outputs 1-3: Reshaped normalized results (same data, different views)
    norm_reshaped = norm_output.view(batch_size, seq_len, hidden_size)[0]  # Take first batch
    out1 = norm_reshaped.view(seq_len, hidden_size)
    out2 = norm_reshaped.view(seq_len, hidden_size).clone()  # Clone to ensure separate tensors
    out3 = norm_reshaped.view(seq_len, hidden_size).clone()
    
    # Output 4: Final addition result
    out4 = final_result
    
    # Output 5: Weight gradients as float32
    out5 = grad_weight
    
    return (out1, out2, out3, out4, out5)