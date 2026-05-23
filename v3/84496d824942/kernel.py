"""
Fused RMS norm kernel implementation with attention norm pattern.
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
    key=['N']
)
@triton.jit
def fused_rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M, N,
    eps,
    input_stride_0,
    input_stride_1,
    output_stride_0,
    output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current row
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    # Calculate row offsets
    input_row_start = row_idx * input_stride_0
    output_row_start = row_idx * output_stride_0
    
    # Load input values for this row in chunks
    mean_square = tl.zeros((), dtype=tl.float32)
    
    # First pass: compute mean square
    for block_start in tl.range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        input_ptrs = input_ptr + input_row_start + offsets * input_stride_1
        x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        mean_square += tl.sum(x * x)
    
    # Compute RMS normalization factor
    mean_square = mean_square / N
    rms_norm_factor = tl.rsqrt(mean_square + eps)
    
    # Second pass: normalize and store
    for block_start in tl.range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load input
        input_ptrs = input_ptr + input_row_start + offsets * input_stride_1
        x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Load weight
        weight_ptrs = weight_ptr + offsets
        w = tl.load(weight_ptrs, mask=mask, other=1.0).to(tl.float32)
        
        # Apply RMS normalization with weight
        normalized = x * rms_norm_factor * w
        
        # Store result
        output_ptrs = output_ptr + output_row_start + offsets * output_stride_1
        tl.store(output_ptrs, normalized.to(output_ptr.dtype.element_ty), mask=mask)


def kernel_function(getitem_1614, add_61):
    """
    Fused RMS norm kernel that implements the attention norm pattern.
    
    This kernel fuses the following operations:
    1. View/clone operations on getitem_1614 to extract weights
    2. RMS normalization of add_61 using extracted weights
    3. Reshaping outputs (handled by returning same tensor 3 times)
    
    Args:
        getitem_1614: Strided tensor [8, 512] -> weights after view/clone
        add_61: Input tensor [1, 8192, 4096] to be normalized
        
    Returns:
        Tuple of 3 identical normalized tensors shaped [8192, 4096]
    """
    # Validate inputs
    assert getitem_1614.device == add_61.device, "Input tensors must be on same device"
    assert getitem_1614.dtype == add_61.dtype, "Input tensors must have same dtype"
    
    device = add_61.device
    dtype = add_61.dtype
    
    # Extract dimensions
    batch_size, seq_len, hidden_dim = add_61.shape
    assert batch_size == 1, "Expected batch size of 1"
    
    # Process weight tensor: view -> clone -> view to get [4096] shape
    # Following the exact pattern from the reference implementation
    weight_viewed = getitem_1614.view(dtype)  # This is a no-op view to bfloat16
    weight_cloned = weight_viewed.clone(memory_format=torch.contiguous_format)
    weight_final = weight_cloned.view(hidden_dim)  # [4096]
    
    # Reshape input to 2D for processing: [1, 8192, 4096] -> [8192, 4096]
    input_2d = add_61.view(seq_len, hidden_dim)
    
    # Allocate output tensor
    output = torch.empty((seq_len, hidden_dim), device=device, dtype=dtype)
    
    # Launch kernel
    M, N = input_2d.shape
    eps = 1e-5
    
    grid = (M,)
    
    fused_rms_norm_kernel[grid](
        input_2d,
        weight_final,
        output,
        M, N,
        eps,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        output.stride(1),
    )
    
    # Return 3 copies as required by the reference implementation
    # All three reshape operations in the reference produce identical [8192, 4096] tensors
    return (output, output, output)