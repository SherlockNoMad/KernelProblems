import torch
import triton
import triton.language as tl


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
    row_idx = tl.program_id(0)
    
    if row_idx >= batch_size:
        return
    
    target = tl.load(targets_ptr + row_idx)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    if target == -100:
        for start_col in range(0, vocab_size, BLOCK_SIZE):
            cols = start_col + col_offsets
            mask = cols < vocab_size
            output_ptrs = output_ptr + cols * output_stride_0 + row_idx * output_stride_1
            tl.store(output_ptrs, tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16), mask=mask)
        return
    
    # Online softmax: single pass to compute max and sum_exp simultaneously
    max_val = -float('inf')
    sum_exp = 0.0
    
    for start_col in range(0, vocab_size, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < vocab_size
        
        logits_ptrs = logits_ptr + row_idx * logits_stride_0 + cols * logits_stride_1
        logits_block = tl.load(logits_ptrs, mask=mask, other=-float('inf'))
        logits_f32 = logits_block.to(tl.float32)
        
        block_max = tl.max(logits_f32)
        new_max = tl.maximum(max_val, block_max)
        
        # Rescale previous sum_exp to new max
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        
        # Add new exponentials with new max
        exp_vals = tl.exp(logits_f32 - new_max)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals)
        
        max_val = new_max
    
    log_sum_exp = tl.log(sum_exp) + max_val
    
    scale_factor = 1.0 / batch_size
    
    # Second pass: compute gradients and store
    for start_col in range(0, vocab_size, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < vocab_size
        
        logits_ptrs = logits_ptr + row_idx * logits_stride_0 + cols * logits_stride_1
        logits_block = tl.load(logits_ptrs, mask=mask, other=0.0)
        logits_f32 = logits_block.to(tl.float32)
        
        log_probs = logits_f32 - log_sum_exp
        probs = tl.exp(log_probs)
        
        target_mask = (cols == target)
        gradients = (probs - target_mask.to(tl.float32)) * scale_factor
        
        gradients_bf16 = gradients.to(tl.bfloat16)
        
        output_ptrs = output_ptr + cols * output_stride_0 + row_idx * output_stride_1
        tl.store(output_ptrs, gradients_bf16, mask=mask)


def kernel_function(target_indices, logits):
    device = logits.device
    batch_size, vocab_size = logits.shape
    
    targets_flat = target_indices.reshape(batch_size)
    
    output = torch.empty((vocab_size, batch_size), dtype=torch.bfloat16, device=device)
    
    # Choose BLOCK_SIZE based on vocab_size for good occupancy
    BLOCK_SIZE = 4096
    
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
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )
    
    return output