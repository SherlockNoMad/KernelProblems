import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    add_ptr, mm_ptr, weight_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: tmp = add_58 + mm_209; out = rms_norm(tmp, weight, eps).
    M = number of rows (8192), N = hidden dim (4096).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    a = tl.load(add_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    # RMS norm
    sq_sum = tl.sum(x * x, axis=0)
    mean_sq = sq_sum / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr, dst_ptr,
    rows, cols, src_row_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy src[rows, cols] (non-contig) to contiguous dst.
    Element index i -> row = i // cols, col = i % cols.
    src offset = row * src_row_stride + col.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    row = offs // cols
    col = offs % cols
    src_off = row * src_row_stride + col

    v = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offs, v, mask=mask)


def kernel_function(mm_209, add_58, wait_tensor_963):
    """
    Fused stages:
      1. _add_rmsnorm_kernel: computes add_58 + mm_209 then RMS-norm with weight chunk 0.
      2. _strided_copy_kernel: copies each split chunk from the [8, N] strided view
         into contiguous tensors with the requested final shapes.

    Outputs match the reference Model: views of the RMS-normed tensor (3 copies)
    plus the reshaped contiguous chunks 1..7 and chunk 7 again (clone_default_7).
    """
    assert mm_209.is_cuda and add_58.is_cuda and wait_tensor_963.is_cuda
    assert mm_209.dtype == torch.bfloat16
    assert add_58.dtype == torch.bfloat16
    assert wait_tensor_963.dtype == torch.bfloat16

    # Shapes
    M, N = 8192, 4096
    assert mm_209.shape == (M, N)
    assert add_58.shape == (1, M, N)
    assert wait_tensor_963.numel() == 218112000

    # View wait_tensor as [8, 27264000]
    total = wait_tensor_963.numel()
    rows = 8
    row_len = total // rows  # 27264000
    wt_view = wait_tensor_963.view(rows, row_len)
    src_row_stride = wt_view.stride(0)  # = row_len

    split_sizes = [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    # offsets along dim 1
    offsets = [0]
    for s in split_sizes:
        offsets.append(offsets[-1] + s)

    # --- Chunk 0: weight for RMS norm, shape [4096] ---
    weight = torch.empty(4096, dtype=torch.bfloat16, device=wait_tensor_963.device)
    size0 = split_sizes[0]  # 512
    n_el0 = rows * size0     # 4096
    src0_ptr = wt_view[:, offsets[0]:offsets[0] + size0]
    BLOCK = 1024
    grid0 = (triton.cdiv(n_el0, BLOCK),)
    _strided_copy_kernel[grid0](
        src0_ptr, weight,
        rows, size0, src_row_stride,
        n_el0, BLOCK_SIZE=BLOCK,
    )

    # --- Fused add + RMS norm ---
    # add_58 is [1, 8192, 4096]; treat as [M, N] (contiguous)
    add58_flat = add_58.view(M, N)
    rms_out = torch.empty((M, N), dtype=torch.bfloat16, device=mm_209.device)

    # Need power-of-2 BLOCK_SIZE >= N
    BLOCK_N = 4096
    grid_rms = (M,)
    _add_rmsnorm_kernel[grid_rms](
        add58_flat, mm_209, weight, rms_out,
        M, N, 1e-5,
        BLOCK_SIZE=BLOCK_N,
    )

    view_default_1 = rms_out.view(M, N)
    view_default_2 = rms_out.view(M, N)
    view_default_3 = rms_out.view(M, N)

    # --- Chunks 1..7 as separate contiguous tensors ---
    # Shapes per _unsafe_view:
    # chunk1: 8*2097152 = 16777216 -> [4096, 4096]
    # chunk2: 8*524288 = 4194304 -> [1024, 4096]
    # chunk3: 8*524288 = 4194304 -> [1024, 4096]
    # chunk4: 8*2097152 = 16777216 -> [4096, 4096]
    # chunk5: 8*512 = 4096 -> [4096]
    # chunk6: 8*7340032 = 58720256 -> [14336, 4096]
    # chunk7: 8*7340032 = 58720256 -> clone_default_6 (shape [8, 7340032])
    # chunk8: 8*7340032 -> clone_default_7 (shape [8, 7340032])

    out_shapes = {
        1: (4096, 4096),
        2: (1024, 4096),
        3: (1024, 4096),
        4: (4096, 4096),
        5: (4096,),
        6: (14336, 4096),
    }

    chunk_outs = {}
    for idx in [1, 2, 3, 4, 5, 6]:
        size_i = split_sizes[idx]
        n_el = rows * size_i
        buf = torch.empty(n_el, dtype=torch.bfloat16, device=wait_tensor_963.device)
        src_slice = wt_view[:, offsets[idx]:offsets[idx] + size_i]
        grid_i = (triton.cdiv(n_el, BLOCK),)
        _strided_copy_kernel[grid_i](
            src_slice, buf,
            rows, size_i, src_row_stride,
            n_el, BLOCK_SIZE=BLOCK,
        )
        chunk_outs[idx] = buf.view(*out_shapes[idx])

    # clone_default_7 is the contiguous clone of chunk 7 viewed as bf16,
    # shape [8, 7340032] (no _unsafe_view applied).
    size7 = split_sizes[7]
    n_el7 = rows * size7
    clone_default_7 = torch.empty((rows, size7), dtype=torch.bfloat16, device=wait_tensor_963.device)
    src_slice7 = wt_view[:, offsets[7]:offsets[7] + size7]
    grid7 = (triton.cdiv(n_el7, BLOCK),)
    _strided_copy_kernel[grid7](
        src_slice7, clone_default_7,
        rows, size7, src_row_stride,
        n_el7, BLOCK_SIZE=BLOCK,
    )

    return (
        view_default_1,
        view_default_2,
        view_default_3,
        chunk_outs[1],
        chunk_outs[2],
        chunk_outs[3],
        chunk_outs[4],
        chunk_outs[5],
        chunk_outs[6],
        clone_default_7,
    )