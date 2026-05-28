import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    add_ptr, mm_ptr, weight_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    a = tl.load(add_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    sq_sum = tl.sum(x * x, axis=0)
    mean_sq = sq_sum / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _contig_copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    v = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, v, mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr, dst_ptr,
    cols, src_row_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    row = offs // cols
    col = offs % cols
    src_off = row * src_row_stride + col

    v = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offs, v, mask=mask)


def kernel_function(mm_209, add_58, wait_tensor_963):
    assert mm_209.is_cuda and add_58.is_cuda and wait_tensor_963.is_cuda

    M, N = 8192, 4096
    total = wait_tensor_963.numel()
    rows = 8
    row_len = total // rows  # 27264000
    wt_view = wait_tensor_963.view(rows, row_len)
    src_row_stride = wt_view.stride(0)

    split_sizes = [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    offsets = [0]
    for s in split_sizes:
        offsets.append(offsets[-1] + s)

    device = wait_tensor_963.device

    # Chunk 0: weight (small, 4096 elements)
    weight = torch.empty(4096, dtype=torch.bfloat16, device=device)
    size0 = split_sizes[0]
    n_el0 = rows * size0
    src0 = wt_view[:, offsets[0]:offsets[0] + size0]
    BLOCK_SMALL = 1024
    _strided_copy_kernel[(triton.cdiv(n_el0, BLOCK_SMALL),)](
        src0, weight, size0, src_row_stride, n_el0, BLOCK_SIZE=BLOCK_SMALL,
    )

    # Fused add + RMSNorm
    add58_flat = add_58.view(M, N)
    rms_out = torch.empty((M, N), dtype=torch.bfloat16, device=mm_209.device)

    _add_rmsnorm_kernel[(M,)](
        add58_flat, mm_209, weight, rms_out,
        M, N, 1e-5,
        BLOCK_SIZE=4096,
        num_warps=8,
        num_stages=3,
    )

    view_default_1 = rms_out.view(M, N)
    view_default_2 = rms_out.view(M, N)
    view_default_3 = rms_out.view(M, N)

    out_shapes = {
        1: (4096, 4096),
        2: (1024, 4096),
        3: (1024, 4096),
        4: (4096, 4096),
        5: (4096,),
        6: (14336, 4096),
    }

    BLOCK = 4096
    chunk_outs = {}
    for idx in [1, 2, 3, 4, 5, 6]:
        size_i = split_sizes[idx]
        n_el = rows * size_i
        buf = torch.empty(n_el, dtype=torch.bfloat16, device=device)
        src_slice = wt_view[:, offsets[idx]:offsets[idx] + size_i]
        # If chunk is contiguous in memory (single row's worth fits exactly), use contig copy
        # But it's strided across 8 rows, so use strided
        _strided_copy_kernel[(triton.cdiv(n_el, BLOCK),)](
            src_slice, buf, size_i, src_row_stride, n_el,
            BLOCK_SIZE=BLOCK, num_warps=4,
        )
        chunk_outs[idx] = buf.view(*out_shapes[idx])

    size7 = split_sizes[7]
    n_el7 = rows * size7
    clone_default_7 = torch.empty((rows, size7), dtype=torch.bfloat16, device=device)
    src_slice7 = wt_view[:, offsets[7]:offsets[7] + size7]
    _strided_copy_kernel[(triton.cdiv(n_el7, BLOCK),)](
        src_slice7, clone_default_7, size7, src_row_stride, n_el7,
        BLOCK_SIZE=BLOCK, num_warps=4,
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