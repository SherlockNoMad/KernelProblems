"""
Fused kernel for transformer feed-forward layer operations:
- Forward: silu -> mul -> mul
- Backward: mul -> silu_backward
This fuses all operations into a single kernel pass for optimal performance.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_feedforward_kernel(
    # Input pointers
    mm_4_ptr, mm_5_ptr, mm_662_ptr,
    # Output pointers  
    out1_ptr, out2_ptr, out3_ptr,
    # Tensor dimensions
    N,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel implementing:
    1. Forward pass: silu(mm_4) * mm_5 -> out1
    2. Intermediate: mm_662 * silu(mm_4) -> out2 
    3. Backward pass: silu_backward(mm_662 * mm_5, mm_4) -> out3
    
    All operations fused in single kernel for maximum efficiency.
    """
    # Calculate the starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input data with masking for boundary conditions
    mm_4_data = tl.load(mm_4_ptr + offsets, mask=mask, other=0.0)
    mm_5_data = tl.load(mm_5_ptr + offsets, mask=mask, other=0.0)
    mm_662_data = tl.load(mm_662_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for computation precision
    mm_4_f32 = mm_4_data.to(tl.float32)
    mm_5_f32 = mm_5_data.to(tl.float32)
    mm_662_f32 = mm_662_data.to(tl.float32)
    
    # Stage 1: Compute SiLU activation - silu(x) = x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_mm_4 = 1.0 / (1.0 + tl.exp(-mm_4_f32))
    silu_mm_4 = mm_4_f32 * sigmoid_mm_4
    
    # Stage 2: Forward computations
    # out1 = silu(mm_4) * mm_5 (corresponds to reshape_default_2)
    out1_f32 = silu_mm_4 * mm_5_f32
    
    # out2 = mm_662 * silu(mm_4) (corresponds to t_default after reshape)
    out2_f32 = mm_662_f32 * silu_mm_4
    
    # Stage 3: Backward computation - SiLU backward gradient
    # silu_backward(grad_output, input) = grad_output * (sigmoid(input) * (1 + input * (1 - sigmoid(input))))
    # grad_output = mm_662 * mm_5, input = mm_4
    grad_output = mm_662_f32 * mm_5_f32
    silu_backward_grad = sigmoid_mm_4 * (1.0 + mm_4_f32 * (1.0 - sigmoid_mm_4))
    out3_f32 = grad_output * silu_backward_grad
    
    # Store results, converting back to original dtype
    tl.store(out1_ptr + offsets, out1_f32.to(mm_4_ptr.dtype.element_ty), mask=mask)
    tl.store(out2_ptr + offsets, out2_f32.to(mm_4_ptr.dtype.element_ty), mask=mask)  
    tl.store(out3_ptr + offsets, out3_f32.to(mm_4_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_4, mm_5, mm_662):
    """
    Wrapper function that launches the fused feed-forward kernel.
    
    Implements the exact computation flow:
    - Forward: silu -> mul -> mul  
    - Backward: mul -> silu_backward
    All fused into single kernel pass.
    """
    # Validate inputs
    assert mm_4.shape == mm_5.shape == mm_662.shape, "Input tensors must have same shape"
    assert mm_4.device == mm_5.device == mm_662.device, "All tensors must be on same device"
    assert mm_4.dtype == mm_5.dtype == mm_662.dtype, "All tensors must have same dtype"
    
    # Get tensor properties
    shape = mm_4.shape
    dtype = mm_4.dtype
    device = mm_4.device
    N = mm_4.numel()
    
    # Allocate output tensors
    # out1: [8192, 14336] - corresponds to reshape_default_2
    out1 = torch.empty(shape, dtype=dtype, device=device)
    
    # out2: [14336, 8192] - corresponds to t_default (transposed)
    out2 = torch.empty((shape[1], shape[0]), dtype=dtype, device=device)
    
    # out3: [14336, 8192] - corresponds to t_default_1 (transposed)  
    out3 = torch.empty((shape[1], shape[0]), dtype=dtype, device=device)
    
    # Create temporary tensors for intermediate results
    temp_out2 = torch.empty(shape, dtype=dtype, device=device)
    temp_out3 = torch.empty(shape, dtype=dtype, device=device)
    
    # Launch kernel with appropriate grid size
    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE']),)
    
    fused_feedforward_kernel[grid](
        mm_4, mm_5, mm_662,
        out1, temp_out2, temp_out3,
        N,
    )
    
    # Transpose the intermediate results to get final outputs
    # This is the only PyTorch operation needed for the reshape/transpose
    out2.copy_(temp_out2.t())
    out3.copy_(temp_out3.t())
    
    return (out1, out2, out3)