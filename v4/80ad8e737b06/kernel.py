"""
Fused cross-entropy loss kernel implementation.
Fuses the entire forward and backward pass into a single efficient kernel.
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
    key=['vocab_size'],
)
@triton.jit
def fused_crossentropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    vocab_size,
    logits_stride_0,
    logits_stride_1,
    output_stride_0,
    output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Log softmax forward pass
    2. Cross-entropy loss computation  
    3. Loss backward pass (gradient computation)
    4. Log softmax backward pass
    
    This eliminates intermediate memory traffic by fusing all operations.
    """
    # Get the current row (batch element)
    row_idx = tl.program_id(0)
    
    if row_idx >= batch_size:
        return
    
    # Load target for this batch element
    target = tl.load(targets_ptr + row_idx)
    
    # Skip ignored targets (-100)
    if target == -100:
        # Zero out the entire row for ignored targets
        col_offsets = tl.arange(0, BLOCK_SIZE)
        for start_col in range(0, vocab_size, BLOCK_SIZE):
            cols = start_col + col_offsets
            mask = cols < vocab_size
            
            output_ptrs = output_ptr + row_idx * output_stride_0 + cols * output_stride_1
            tl.store(output_ptrs, 0.0, mask=mask)
        return
    
    # Process logits in blocks for memory efficiency
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First pass: find maximum for numerical stability
    max_val = -float('inf')
    for start_col in range(0, vocab_size, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < vocab_size
        
        logits_ptrs = logits_ptr + row_idx * logits_stride_0 + cols * logits_stride_1
        logits_block = tl.load(logits_ptrs, mask=mask, other=-float('inf'))
        logits_f32 = logits_block.to(tl.float32)
        
        block_max = tl.max(logits_f32)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exponentials
    sum_exp = 0.0
    for start_col in range(0, vocab_size, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < vocab_size
        
        logits_ptrs = logits_ptr + row_idx * logits_stride_0 + cols * logits_stride_1
        logits_block = tl.load(logits_ptrs, mask=mask, other=0.0)
        logits_f32 = logits_block.to(tl.float32)
        
        exp_vals = tl.exp(logits_f32 - max_val)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals)
    
    # Compute log_sum_exp for softmax normalization
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Load the target logit for loss computation
    target_logit_ptr = logits_ptr + row_idx * logits_stride_0 + target * logits_stride_1
    target_logit = tl.load(target_logit_ptr).to(tl.float32)
    target_log_prob = target_logit - log_sum_exp
    
    # Third pass: compute gradients and store results
    # Gradient = (softmax_prob - target_indicator) / batch_size
    scale_factor = 1.0 / batch_size
    
    for start_col in range(0, vocab_size, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < vocab_size
        
        # Load logits for this block
        logits_ptrs = logits_ptr + row_idx * logits_stride_0 + cols * logits_stride_1
        logits_block = tl.load(logits_ptrs, mask=mask, other=0.0)
        logits_f32 = logits_block.to(tl.float32)
        
        # Compute log probabilities
        log_probs = logits_f32 - log_sum_exp
        
        # Compute softmax probabilities
        probs = tl.exp(log_probs)
        
        # Compute gradients: prob - target_indicator
        target_mask = (cols == target)
        gradients = probs - target_mask.to(tl.float32)
        
        # Scale by batch size (from loss backward pass)
        gradients = gradients * scale_factor
        
        # Convert to bfloat16 and store
        gradients_bf16 = gradients.to(tl.bfloat16)
        
        # Store to transposed output (note the swapped indices for transpose)
        output_ptrs = output_ptr + cols * output_stride_0 + row_idx * output_stride_1
        tl.store(output_ptrs, gradients_bf16, mask=mask)


def kernel_function(target_indices, logits):
    """
    Fused cross-entropy loss kernel that computes gradients efficiently.
    
    This kernel fuses the entire computation chain:
    - Forward: logits -> log_softmax -> nll_loss  
    - Backward: loss_grad -> nll_loss_backward -> log_softmax_backward
    - Plus all the reshaping and type conversions
    
    Args:
        target_indices: [1, batch_size] tensor of target class indices
        logits: [batch_size, vocab_size] tensor of logits
        
    Returns:
        [vocab_size, batch_size] tensor of gradients (transposed)
    """
    # Input validation
    assert target_indices.device == logits.device, "Device mismatch"
    assert target_indices.dtype == torch.int64, f"Expected int64 targets, got {target_indices.dtype}"
    assert logits.dtype == torch.bfloat16, f"Expected bfloat16 logits, got {logits.dtype}"
    
    device = logits.device
    batch_size, vocab_size = logits.shape
    
    # Flatten target indices to [batch_size]
    targets_flat = target_indices.reshape(batch_size)
    
    # Allocate output tensor (transposed shape)
    output = torch.empty((vocab_size, batch_size), dtype=torch.bfloat16, device=device)
    
    # Launch kernel with one thread block per batch element
    grid = (batch_size,)
    
    fused_crossentropy_kernel[grid](
        logits,
        targets_flat, 
        output,
        batch_size,
        vocab_size,
        logits.stride(0),
        logits.stride(1), 
        output.stride(0),
        output.stride(1),
    )
    
    return output