import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_bwd_fused_kernel(
    mm672_ptr, view1794_ptr, mm674_ptr, emb_ptr, rstd_ptr, w_ptr, add220_ptr,
    out_ptr, gw_partial_ptr,
    M, N, NUM_ROW_BLOCKS,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_off = row * N

    g1 = tl.load(view1794_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(mm672_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    g3 = tl.load(mm674_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    s1 = (g1 + g2).to(tl.bfloat16)
    grad_y_bf16 = (s1.to(tl.float32) + g3).to(tl.bfloat16)
    grad_y = grad_y_bf16.to(tl.float32)

    x = tl.load(emb_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    xhat = x * rstd
    g = grad_y * w

    gw_partial = grad_y * xhat
    tl.store(gw_partial_ptr + row_off + cols, gw_partial, mask=mask)

    c = tl.sum(g * xhat, axis=0) / N

    grad_x = rstd * (g - xhat * c)

    add220 = tl.load(add220_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    out = add220 + grad_x

    tl.store(out_ptr + row_off + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _grad_w_reduce_kernel(
    gw_partial_in_ptr, gw_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r_start in range(0, M, BLOCK_M):
        rows = r_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        offs = rows[:, None] * N + cols[None, :]
        m = row_mask[:, None] & col_mask[None, :]
        v = tl.load(gw_partial_in_ptr + offs, mask=m, other=0.0)
        acc += tl.sum(v, axis=0)

    tl.store(gw_ptr + cols, acc, mask=col_mask)


@triton.jit
def _grad_w_partial_split_kernel(
    gw_partial_in_ptr, partial_out_ptr,
    M, N, ROWS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    row_start = pid_m * ROWS_PER_BLOCK
    row_end = tl.minimum(row_start + ROWS_PER_BLOCK, M)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r_start in range(row_start, row_end, BLOCK_M):
        rows = r_start + tl.arange(0, BLOCK_M)
        row_mask = rows < row_end
        offs = rows[:, None] * N + cols[None, :]
        m = row_mask[:, None] & col_mask[None, :]
        v = tl.load(gw_partial_in_ptr + offs, mask=m, other=0.0)
        acc += tl.sum(v, axis=0)

    tl.store(partial_out_ptr + pid_m * N + cols, acc, mask=col_mask)


@triton.jit
def _grad_w_final_reduce_kernel(
    partial_ptr, gw_ptr,
    N,
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS_C: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(0, NUM_BLOCKS_C):
        v = tl.load(partial_ptr + i * N + cols, mask=col_mask, other=0.0)
        acc += v

    tl.store(gw_ptr + cols, acc, mask=col_mask)


def kernel_function(mm_672, view_1794, mm_674, embedding, getitem_1_recomputed, _unsafe_view_432, add_220):
    assert mm_672.is_cuda
    M = 8192
    N = 4096

    mm672_f = mm_672.contiguous().view(M, N)
    view1794_f = view_1794.contiguous().view(M, N)
    mm674_f = mm_674.contiguous().view(M, N)
    emb_f = embedding.contiguous().view(M, N)
    add220_f = add_220.contiguous().view(M, N)
    rstd_f = getitem_1_recomputed.contiguous().view(M)
    w_f = _unsafe_view_432.contiguous()

    out0 = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_672.device)
    out0_f = out0.view(M, N)

    gw_partial = torch.empty((M, N), dtype=torch.float32, device=mm_672.device)

    _rmsnorm_bwd_fused_kernel[(M,)](
        mm672_f, view1794_f, mm674_f, emb_f, rstd_f, w_f, add220_f,
        out0_f, gw_partial,
        M, N, 1,
        BLOCK_N=4096,
        num_warps=8,
    )

    # Two-stage reduction with high parallelism
    BLOCK_N_RED = 32
    BLOCK_M_RED = 32
    ROWS_PER_BLOCK = 512  # M / 512 = 16 row-blocks; total grid = 128 * 16 = 2048 blocks
    NUM_ROW_BLOCKS = triton.cdiv(M, ROWS_PER_BLOCK)
    NUM_COL_BLOCKS = triton.cdiv(N, BLOCK_N_RED)

    stage1_out = torch.empty((NUM_ROW_BLOCKS, N), dtype=torch.float32, device=mm_672.device)
    _grad_w_partial_split_kernel[(NUM_COL_BLOCKS, NUM_ROW_BLOCKS)](
        gw_partial, stage1_out,
        M, N, ROWS_PER_BLOCK,
        BLOCK_M=BLOCK_M_RED,
        BLOCK_N=BLOCK_N_RED,
        num_warps=1,
    )

    grad_w = torch.empty((N,), dtype=torch.float32, device=mm_672.device)
    BLOCK_N_FINAL = 64
    grid_final = (triton.cdiv(N, BLOCK_N_FINAL),)
    _grad_w_final_reduce_kernel[grid_final](
        stage1_out, grad_w,
        N,
        BLOCK_N=BLOCK_N_FINAL,
        NUM_BLOCKS_C=NUM_ROW_BLOCKS,
        num_warps=2,
    )

    return (out0, grad_w)