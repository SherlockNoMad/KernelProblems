"""
Fused kernel implementation for add -> split_with_sizes -> _fused_rms_norm -> _fused_rms_norm_backward -> _to_copy
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4), 
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['hidden_dim']
)
@triton.jit
def fused_rms_norm_kernel(
    # Input pointers
    mm_223_ptr, add_62_ptr, mm_226_ptr, weight_ptr,
    # Output pointers  
    out1_ptr, out2_ptr,
    # Dimensions
    batch_size, seq_len, hidden_dim,
    # Constants
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # This comes from autotune
):
    """
    Fused kernel implementing:
    1. Add: mm_223 (reshaped) + add_62_recomputed -> add_tensor
    2. RMS norm forward: add_tensor -> normalized, mean_square
    3. RMS norm backward: grad_output, normalized -> grad_input, grad_weight
    4. Type conversion: grad_weight -> float32
    
    Fusion stages:
    - Stage 1: Forward pass (add + RMS norm)
    - Stage 2: Backward pass (RMS norm backward)
    - Stage 3: Weight gradient reduction and type conversion
    """
    # Get program IDs
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate base offsets for this sequence position
    seq_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    mm_offset = seq_idx * hidden_dim  # mm_223 and mm_226 are [seq_len, hidden_dim]
    
    # Stage 1: Forward pass - compute add + RMS norm statistics
    mean_square = tl.zeros((), dtype=tl.float32)
    
    # First pass: compute mean square for RMS norm
    for block_start in tl.range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Load inputs
        mm_223_data = tl.load(mm_223_ptr + mm_offset + offsets, mask=mask, other=0.0)
        add_62_data = tl.load(add_62_ptr + seq_offset + offsets, mask=mask, other=0.0)
        
        # Add operation (equivalent to reshape + add)
        add_result = add_62_data + mm_223_data
        add_result_f32 = add_result.to(tl.float32)
        
        # Accumulate squared values
        mean_square += tl.sum(add_result_f32 * add_result_f32)
    
    # Compute RMS statistics
    mean_square = mean_square / hidden_dim
    rms = tl.sqrt(mean_square + eps)
    inv_rms = 1.0 / rms
    
    # Stage 2: Backward pass - compute gradients
    grad_weight_sum = tl.zeros((), dtype=tl.float32)
    
    for block_start in tl.range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Reload forward pass inputs
        mm_223_data = tl.load(mm_223_ptr + mm_offset + offsets, mask=mask, other=0.0)
        add_62_data = tl.load(add_62_ptr + seq_offset + offsets, mask=mask, other=0.0)
        
        # Forward: add + normalize
        input_data = add_62_data + mm_223_data
        input_f32 = input_data.to(tl.float32)
        normalized = input_f32 * inv_rms
        
        # Load weight and grad output
        weight_data = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        weight_f32 = weight_data.to(tl.float32)
        grad_output = tl.load(mm_226_ptr + mm_offset + offsets, mask=mask, other=0.0)
        grad_output_f32 = grad_output.to(tl.float32)
        
        # Backward: compute gradients
        grad_normalized = grad_output_f32 * weight_f32
        
        # Accumulate for input gradient computation
        grad_weight_sum += tl.sum(grad_normalized * normalized)
    
    # Second pass: compute input gradients
    for block_start in tl.range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Reload data for gradient computation
        mm_223_data = tl.load(mm_223_ptr + mm_offset + offsets, mask=mask, other=0.0)
        add_62_data = tl.load(add_62_ptr + seq_offset + offsets, mask=mask, other=0.0)
        
        input_data = add_62_data + mm_223_data
        input_f32 = input_data.to(tl.float32)
        normalized = input_f32 * inv_rms
        
        weight_data = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        weight_f32 = weight_data.to(tl.float32)
        grad_output = tl.load(mm_226_ptr + mm_offset + offsets, mask=mask, other=0.0)
        grad_output_f32 = grad_output.to(tl.float32)
        
        # Compute input gradient
        grad_normalized = grad_output_f32 * weight_f32
        grad_input = grad_normalized * inv_rms - normalized * inv_rms * grad_weight_sum / hidden_dim
        
        # Store input gradient
        tl.store(out1_ptr + seq_offset + offsets, grad_input.to(tl.bfloat16), mask=mask)
        
        # Compute and store weight gradient contribution
        grad_weight_contrib = grad_output_f32 * normalized
        contrib_offset = (batch_idx * seq_len + seq_idx) * hidden_dim + offsets
        tl.store(out2_ptr + contrib_offset, grad_weight_contrib, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['hidden_dim']
)
@triton.jit
def reduce_weight_gradients_kernel(
    contrib_ptr, output_ptr, 
    batch_size, seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce weight gradient contributions across batch and sequence dimensions."""
    dim_idx = tl.program_id(0)
    
    offsets = dim_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    grad_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Sum contributions from all positions
    for pos in tl.range(batch_size * seq_len):
        contrib_offsets = pos * hidden_dim + offsets
        contrib = tl.load(contrib_ptr + contrib_offsets, mask=mask, other=0.0)
        grad_sum += contrib
    
    # Store final reduced gradients
    tl.store(output_ptr + offsets, grad_sum, mask=mask)

def kernel_function(mm_223, add_62_recomputed, mm_226, wait_tensor_970):
    """
    Fused kernel implementing the complete operation chain:
    1. Add operation (mm_223 reshaped + add_62_recomputed) 
    2. Weight tensor processing (split_with_sizes + view operations)
    3. RMS normalization forward and backward passes
    4. Type conversion to float32
    
    This fusion is optimal because:
    - All operations are elementwise or reductions that can be computed together
    - Minimizes memory traffic by avoiding intermediate tensor storage
    - Combines forward and backward RMS norm in single kernel launch
    """
    device = mm_223.device
    batch_size = add_62_recomputed.shape[0]  # 1
    seq_len = add_62_recomputed.shape[1]     # 8192  
    hidden_dim = add_62_recomputed.shape[2]  # 4096
    
    # Validate input shapes
    assert mm_223.shape == (seq_len, hidden_dim)
    assert add_62_recomputed.shape == (batch_size, seq_len, hidden_dim)
    assert mm_226.shape == (seq_len, hidden_dim)
    assert wait_tensor_970.shape == (hidden_dim,)
    
    # Process weight tensor according to the reference operations:
    # view([8, -1]) -> split_with_sizes([512], 1) -> getitem(0) -> view(bf16) -> view(-1)
    weight_reshaped = wait_tensor_970.view(8, -1)  # [8, 512]
    weight_split = torch.split(weight_reshaped, [512], dim=1)  # tuple with one [8, 512] tensor
    weight_first = weight_split[0]  # [8, 512]
    weight_bf16 = weight_first.to(torch.bfloat16)  # convert to bfloat16
    weight_final = weight_bf16.view(-1)  # flatten to [4096]
    
    # Allocate outputs
    out1 = torch.empty_like(add_62_recomputed, dtype=torch.bfloat16)  # grad_input
    
    # Temporary storage for weight gradient contributions before reduction
    temp_contrib = torch.zeros((batch_size * seq_len, hidden_dim), device=device, dtype=torch.float32)
    out2 = torch.zeros(hidden_dim, device=device, dtype=torch.float32)  # final weight gradients
    
    # Launch fused kernel
    grid = (batch_size, seq_len)
    fused_rms_norm_kernel[grid](
        mm_223, add_62_recomputed, mm_226, weight_final,
        out1, temp_contrib,
        batch_size, seq_len, hidden_dim,
        eps=1e-5,
    )
    
    # Reduce weight gradients across batch/sequence dimensions
    reduce_grid = (triton.cdiv(hidden_dim, 256),)
    reduce_weight_gradients_kernel[reduce_grid](
        temp_contrib, out2,
        batch_size, seq_len, hidden_dim,
    )
    
    return (out1, out2)