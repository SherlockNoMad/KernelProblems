import torch
import triton
import triton.language as tl


@triton.jit
def fused_rms_norm_with_weight_gather_kernel(
    input_ptr,
    weight_src_ptr,
    output_ptr,
    M, N,
    eps,
    input_stride_0,
    output_stride_0,
    weight_src_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    input_row_start = row_idx * input_stride_0
    output_row_start = row_idx * output_stride_0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input row
    x = tl.load(input_ptr + input_row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute RMS normalization factor
    mean_square = tl.sum(x * x, axis=0) / N
    rms_norm_factor = tl.rsqrt(mean_square + eps)
    
    # Load weight with gather pattern: weight_src is [8, 512] flattened as weight[0..4095]
    # where row = offset // 512, col = offset % 512
    w_row = offsets // 512
    w_col = offsets % 512
    w = tl.load(weight_src_ptr + w_row * weight_src_stride_0 + w_col, mask=mask, other=1.0).to(tl.float32)
    
    # Fused: normalize and scale by weight in single pass
    normalized = x * rms_norm_factor * w
    
    tl.store(output_ptr + output_row_start + offsets, normalized.to(tl.bfloat16), mask=mask)


def kernel_function(getitem_1614, add_61):
    device = add_61.device
    dtype = add_61.dtype
    
    # add_61 shape: [batch_size, seq_len, hidden_dim] or could be [1, seq_len, hidden_dim]
    orig_shape = add_61.shape
    hidden_dim = orig_shape[-1]
    
    # Flatten to 2D: [total_rows, hidden_dim]
    input_2d = add_61.reshape(-1, hidden_dim).contiguous()
    M, N = input_2d.shape
    
    output = torch.empty((M, N), device=device, dtype=dtype)
    
    eps = 1e-5
    
    # BLOCK_SIZE must be power of 2 >= N
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    num_warps = 16 if BLOCK_SIZE >= 2048 else (8 if BLOCK_SIZE >= 1024 else 4)
    
    grid = (M,)
    
    fused_rms_norm_with_weight_gather_kernel[grid](
        input_2d,
        getitem_1614,
        output,
        M, N,
        eps,
        input_2d.stride(0),
        output.stride(0),
        getitem_1614.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    
    # Reshape output back to original shape
    output = output.view(orig_shape)
    
    return (output, output, output)