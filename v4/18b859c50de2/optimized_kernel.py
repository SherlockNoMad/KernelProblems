import torch
import triton
import triton.language as tl


@triton.jit
def fused_residual_add_rms_norm_kernel(
    mm_223_ptr,
    add_62_ptr,
    weight_ptr,
    output_ptr,
    M,
    N,
    weight_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start = row_idx * N
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    offsets = row_start + col_offsets

    # Load inputs
    mm_data = tl.load(mm_223_ptr + offsets, mask=mask, other=0.0)
    add_data = tl.load(add_62_ptr + offsets, mask=mask, other=0.0)

    # Residual add in float32
    residual = add_data.to(tl.float32) + mm_data.to(tl.float32)

    # RMS norm
    mean_square = tl.sum(residual * residual) / N
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)

    # Load weight: weight is in a reshaped view [8, X] with given stride
    # First 512 elements per row of the reshaped view = first 4096 elements total = our weight
    chunk_id = col_offsets // 512
    offset_in_chunk = col_offsets % 512
    weight_offsets = chunk_id * weight_stride + offset_in_chunk
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=1.0)

    # Normalize and store
    normalized = (residual * inv_rms) * weight.to(tl.float32)
    tl.store(output_ptr + offsets, normalized.to(tl.bfloat16), mask=mask)


@triton.jit
def strided_copy_kernel(
    src_ptr,
    dst_ptr,
    row_len,
    src_stride,
    dst_stride,
    src_col_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy rows from a strided source to a contiguous destination."""
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    block_start = block_idx * BLOCK_SIZE
    col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < row_len

    # Source is strided: row_idx * src_stride + (src_col_offset + col_offsets)
    src_off = row_idx * src_stride + src_col_offset + col_offsets
    dst_off = row_idx * dst_stride + col_offsets

    data = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_off, data, mask=mask)


def kernel_function(mm_223, add_62, wait_tensor_969):
    device = mm_223.device
    dtype = mm_223.dtype

    M, N = mm_223.shape[0], mm_223.shape[1]
    eps = 1e-5

    wait_reshaped = wait_tensor_969.view(8, -1)  # [8, 65667584]
    weight_stride = wait_reshaped.stride(0)

    # Output 1: fused residual add + RMS norm
    output1 = torch.empty((M, N), device=device, dtype=dtype)

    fused_residual_add_rms_norm_kernel[(M,)](
        mm_223, add_62, wait_reshaped, output1,
        M, N, weight_stride, eps,
        BLOCK_SIZE=4096,
        num_warps=8,
    )

    # Output 2: wait_reshaped[:, 512:] made contiguous, then reshaped to [128256, 4096]
    num_rows = 8
    row_len = wait_reshaped.shape[1] - 512
    part2 = torch.empty((num_rows, row_len), device=device, dtype=dtype)

    grid = (num_rows, triton.cdiv(row_len, 4096))
    strided_copy_kernel[grid](
        wait_tensor_969, part2,
        row_len, weight_stride, row_len, 512,
        BLOCK_SIZE=4096,
        num_warps=8,
    )

    part2 = part2.reshape(128256, 4096)

    return (output1, part2)