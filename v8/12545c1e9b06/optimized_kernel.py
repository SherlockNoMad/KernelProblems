import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_rmsnorm_kernel(
    emb_weight_ptr,
    indices_ptr,
    rms_weight_ptr,
    out_ptr,
    N, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(indices_ptr + row).to(tl.int64)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(emb_weight_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(rms_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x)
    mean_sq = sum_sq / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    y = x * rstd * w
    tl.store(out_ptr + row * D + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr,
    dst_ptr,
    row_stride,
    col_offset,
    cols_per_row,
    total,
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


@triton.jit
def _contig_copy_kernel(
    src_ptr,
    dst_ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


def kernel_function(wait_tensor_871, arg583_1, wait_tensor_873):
    device = wait_tensor_871.device

    emb_weight = wait_tensor_871.view(128256, 4096)

    N_873 = wait_tensor_873.numel()
    row_len = N_873 // 8

    split_sizes = [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    col_offsets = [0]
    for s in split_sizes:
        col_offsets.append(col_offsets[-1] + s)

    def extract_slice(idx, size):
        total = 8 * size
        out = torch.empty(total, dtype=torch.bfloat16, device=device)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(total, BLOCK_SIZE),)
        _strided_copy_kernel[grid](
            wait_tensor_873, out,
            row_len, col_offsets[idx], size, total,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return out

    rms_weight_flat = extract_slice(0, 512)

    indices = arg583_1.reshape(-1)
    N = indices.numel()
    D = 4096

    embed_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)

    grid = (N,)
    _embedding_rmsnorm_kernel[grid](
        emb_weight, indices, rms_weight_flat, embed_out,
        N, D, 1e-5,
        BLOCK_D=4096,
        num_warps=8,
        num_stages=2,
    )

    view_3 = embed_out
    view_4 = embed_out
    view_5 = embed_out

    s1 = extract_slice(1, 2097152).view(4096, 4096)
    s2 = extract_slice(2, 524288).view(1024, 4096)
    s3 = extract_slice(3, 524288).view(1024, 4096)
    s4 = extract_slice(4, 2097152).view(4096, 4096)
    s5 = extract_slice(5, 512)
    s6 = extract_slice(6, 7340032).view(14336, 4096)

    return (view_3, view_4, view_5, s1, s2, s3, s4, s5, s6)