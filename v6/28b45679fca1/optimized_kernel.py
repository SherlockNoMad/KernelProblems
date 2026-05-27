import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    mm_ptr, add_ptr, x_ptr, rstd_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * N + cols

    a = tl.load(mm_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(add_ptr + off, mask=mask, other=0.0).to(tl.float32)
    x = a + b
    tl.store(x_ptr + off, x.to(x_ptr.dtype.element_ty), mask=mask)

    sq = x * x
    var = tl.sum(sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _bwd_dx_dw_fused_kernel(
    go_ptr, x_ptr, w_ptr, rstd_ptr, gi_ptr, partial_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in tl.static_range(BLOCK_ROWS):
        row = row_start + i
        off = row * N + cols
        go = tl.load(go_ptr + off, mask=col_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=col_mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)

        x_hat = x * rstd
        gw_x = go * w
        s = tl.sum(gw_x * x_hat, axis=0) / N
        gi = rstd * (gw_x - x_hat * s)
        tl.store(gi_ptr + off, gi.to(gi_ptr.dtype.element_ty), mask=col_mask)

        acc_dw += go * x_hat

    out_off = pid * N + cols
    tl.store(partial_ptr + out_off, acc_dw, mask=col_mask)


@triton.jit
def _bwd_dw_reduce_2d_kernel(
    partial_ptr, gw_partial_ptr,
    N, NUM_M_BLOCKS, SPLIT_M,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    row_start = pid_m * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = rows < NUM_M_BLOCKS

    mask2d = row_mask[:, None] & col_mask[None, :]
    off2d = rows[:, None] * N + cols[None, :]
    v = tl.load(partial_ptr + off2d, mask=mask2d, other=0.0)
    acc = tl.sum(v, axis=0)

    out_off = pid_m * N + cols
    tl.store(gw_partial_ptr + out_off, acc, mask=col_mask)


@triton.jit
def _bwd_dw_final_kernel(
    gw_partial_ptr, gw_ptr,
    N, SPLIT_M,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, SPLIT_M):
        off = k * N + cols
        v = tl.load(gw_partial_ptr + off, mask=col_mask, other=0.0)
        acc += v

    tl.store(gw_ptr + cols, acc, mask=col_mask)


def kernel_function(mm_223, add_62_recomputed, mm_226, view_3253):
    assert mm_223.is_cuda and add_62_recomputed.is_cuda and mm_226.is_cuda and view_3253.is_cuda
    eps = 1e-5
    M = 8192
    N = 4096

    device = mm_223.device

    mm_223_c = mm_223.contiguous()
    add_62_c = add_62_recomputed.contiguous()
    mm_226_c = mm_226.contiguous()
    w_c = view_3253.contiguous()

    add_tensor = torch.empty((1, M, N), dtype=mm_223.dtype, device=device)
    rstd = torch.empty((M,), dtype=torch.float32, device=device)

    BLOCK_N_FWD = 4096

    _fwd_kernel[(M,)](
        mm_223_c, add_62_c, add_tensor, rstd,
        M, N, eps,
        BLOCK_N=BLOCK_N_FWD,
        num_warps=8,
    )

    grad_input = torch.empty((1, M, N), dtype=mm_223.dtype, device=device)

    BLOCK_ROWS = 16
    NUM_BLOCKS = triton.cdiv(M, BLOCK_ROWS)
    partial = torch.empty((NUM_BLOCKS, N), dtype=torch.float32, device=device)

    _bwd_dx_dw_fused_kernel[(NUM_BLOCKS,)](
        mm_226_c, add_tensor, w_c, rstd, grad_input, partial,
        M, N,
        BLOCK_N=BLOCK_N_FWD,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=8,
    )

    grad_weight = torch.empty((N,), dtype=torch.float32, device=device)

    # 2D tiling: split across columns and across M-reduction to fully fill SMs
    BLOCK_N_R = 128
    SPLIT_M = 8  # split NUM_BLOCKS (512) into 8 chunks of 64 rows each
    ROWS_PER_BLOCK = triton.cdiv(NUM_BLOCKS, SPLIT_M)

    gw_partial = torch.empty((SPLIT_M, N), dtype=torch.float32, device=device)

    grid_r = (triton.cdiv(N, BLOCK_N_R), SPLIT_M)
    _bwd_dw_reduce_2d_kernel[grid_r](
        partial, gw_partial,
        N, NUM_BLOCKS, SPLIT_M,
        BLOCK_N=BLOCK_N_R,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        num_warps=4,
    )

    BLOCK_N_F = 256
    grid_f = (triton.cdiv(N, BLOCK_N_F),)
    _bwd_dw_final_kernel[grid_f](
        gw_partial, grad_weight,
        N, SPLIT_M,
        BLOCK_N=BLOCK_N_F,
        num_warps=4,
    )

    return (grad_input, grad_weight)