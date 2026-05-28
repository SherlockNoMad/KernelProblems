import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_rmsnorm_kernel(
    emb_weight_ptr,  # [V, D] bf16
    indices_ptr,     # [N] int64
    rms_weight_ptr,  # [D] bf16
    out_ptr,         # [N, D] bf16
    N, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    idx = tl.load(indices_ptr + row).to(tl.int64)

    # Compute sum of squares
    sum_sq = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(emb_weight_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    mean_sq = sum_sq / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Apply normalization and weight
    for d_start in range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(emb_weight_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(rms_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(out_ptr + row * D + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr,       # base of wait_tensor_873 (flat bf16)
    dst_ptr,       # contiguous output
    row_stride,    # stride between rows = 27264000
    col_offset,    # starting column in each row
    cols_per_row,  # number of cols to take per row
    num_rows,      # 8
    total,         # num_rows * cols_per_row
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    row = offs // cols_per_row
    col = offs % cols_per_row
    src_idx = row * row_stride + col_offset + col
    x = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


def kernel_function(wait_tensor_871, arg583_1, wait_tensor_873):
    """
    Fused implementation:
    - Extracts embedding weight from wait_tensor_871 (reshape only, no data movement needed)
    - Extracts strided slices from wait_tensor_873 viewed as [8, 27264000]
    - Fuses embedding lookup + RMS norm into single kernel
    - Returns views/clones as specified
    """
    device = wait_tensor_871.device
    assert wait_tensor_871.dtype == torch.bfloat16
    assert wait_tensor_873.dtype == torch.bfloat16

    # wait_tensor_871 viewed as [8, 65667072], take first split (all of it),
    # then view as [128256, 4096]. Since the entire flat tensor IS the embedding weight,
    # we just reinterpret it.
    emb_weight = wait_tensor_871.view(128256, 4096)

    # wait_tensor_873 viewed as [8, 27264000]
    N_873 = wait_tensor_873.numel()
    row_len = N_873 // 8  # 27264000
    assert row_len * 8 == N_873

    split_sizes = [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    # Compute column offsets
    col_offsets = [0]
    for s in split_sizes:
        col_offsets.append(col_offsets[-1] + s)

    # Helper to extract slice [8, size] -> contiguous flat [8*size]
    def extract_slice(idx, size):
        total = 8 * size
        out = torch.empty(total, dtype=torch.bfloat16, device=device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total, BLOCK_SIZE),)
        _strided_copy_kernel[grid](
            wait_tensor_873, out,
            row_len, col_offsets[idx], size, 8, total,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

    # Extract rms norm weight: split 0, size 512, total 4096
    rms_weight_flat = extract_slice(0, 512)  # shape [4096]

    # arg583_1 is [1, 8192], indices are int64
    indices = arg583_1.reshape(-1)
    N = indices.numel()  # 8192
    D = 4096

    # Output of embedding + rms norm
    embed_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)

    BLOCK_D = 1024
    grid = (N,)
    _embedding_rmsnorm_kernel[grid](
        emb_weight, indices, rms_weight_flat, embed_out,
        N, D, 1e-5,
        BLOCK_D=BLOCK_D,
    )

    # view_default_3, _4, _5 are all the same [8192, 4096] view
    view_3 = embed_out
    view_4 = embed_out
    view_5 = embed_out

    # Extract other slices
    # split 1: size 2097152 -> [4096, 4096]
    s1 = extract_slice(1, 2097152).view(4096, 4096)
    # split 2: size 524288 -> [1024, 4096]
    s2 = extract_slice(2, 524288).view(1024, 4096)
    # split 3: size 524288 -> [1024, 4096]
    s3 = extract_slice(3, 524288).view(1024, 4096)
    # split 4: size 2097152 -> [4096, 4096]
    s4 = extract_slice(4, 2097152).view(4096, 4096)
    # split 5: size 512 -> [4096]
    s5 = extract_slice(5, 512)
    # split 6: size 7340032 -> [14336, 4096]
    s6 = extract_slice(6, 7340032).view(14336, 4096)

    return (view_3, view_4, view_5, s1, s2, s3, s4, s5, s6)