import triton
import triton.language as tl
import torch


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, RSTD_ptr,
    N, D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    sq = x * x
    mean_sq = tl.sum(sq, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    tl.store(RSTD_ptr + row, rstd)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y_ptr + row * D + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_dx_fused_kernel(
    MM230_ptr, MM232_ptr, X_ptr, W_ptr, RSTD_ptr,
    GETITEM420_ptr, ADD_OUT_ptr, GO_OUT_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    a = tl.load(MM230_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(MM232_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    go_bf16 = (a + b).to(GO_OUT_ptr.dtype.element_ty)
    tl.store(GO_OUT_ptr + row * D + cols, go_bf16, mask=mask)
    go = go_bf16.to(tl.float32)

    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    gw = go * w
    sum_gwx = tl.sum(gw * x, axis=0)
    c = sum_gwx * rstd * rstd / D

    dx = rstd * (gw - x * c)

    g420 = tl.load(GETITEM420_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = g420 + dx
    tl.store(ADD_OUT_ptr + row * D + cols, out.to(ADD_OUT_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_dw_partial_kernel(
    GO_ptr, X_ptr, RSTD_ptr, DW_PARTIAL_ptr,
    N, D, N_SPLITS,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)

    col_start = pid_d * BLOCK_D
    cols = col_start + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    rows_per_split = N // N_SPLITS
    n_begin = pid_n * rows_per_split
    n_end = n_begin + rows_per_split

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for n_start in range(n_begin, n_end, BLOCK_N):
        rows = n_start + tl.arange(0, BLOCK_N)
        row_mask = rows < n_end

        offs = rows[:, None] * D + cols[None, :]
        m = row_mask[:, None] & col_mask[None, :]

        go = tl.load(GO_ptr + offs, mask=m, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        contrib = go * x * rstd[:, None]
        acc += tl.sum(contrib, axis=0)

    tl.store(DW_PARTIAL_ptr + pid_n * D + cols, acc, mask=col_mask)


@triton.jit
def _dw_reduce_kernel(
    DW_PARTIAL_ptr, DW_ptr,
    D, N_SPLITS,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(0, N_SPLITS):
        v = tl.load(DW_PARTIAL_ptr + i * D + cols, mask=mask, other=0.0)
        acc += v

    tl.store(DW_ptr + cols, acc, mask=mask)


def kernel_function(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420):
    device = add_62_recomputed.device

    weight_2d = getitem_1624.contiguous()
    weight = weight_2d.reshape(4096)

    N = 8192
    D = 4096

    x = add_62_recomputed.reshape(N, D)

    y = torch.empty((N, D), dtype=torch.bfloat16, device=device)
    rstd = torch.empty((N,), dtype=torch.float32, device=device)

    BLOCK_SIZE = 4096
    _rmsnorm_fwd_kernel[(N,)](
        x, weight, y, rstd,
        N, D, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    add_out = torch.empty((1, 8192, 4096), dtype=torch.bfloat16, device=device)
    g420 = getitem_420.reshape(N, D)
    add_out_2d = add_out.view(N, D)
    go = torch.empty((N, D), dtype=torch.bfloat16, device=device)

    _rmsnorm_bwd_dx_fused_kernel[(N,)](
        mm_230, mm_232, x, weight, rstd,
        g420, add_out_2d, go,
        N, D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    BLOCK_D = 64
    BLOCK_N = 128
    N_SPLITS = 32
    assert N % N_SPLITS == 0

    dw_partial = torch.empty((N_SPLITS, D), dtype=torch.float32, device=device)

    grid_dw = (triton.cdiv(D, BLOCK_D), N_SPLITS)
    _rmsnorm_bwd_dw_partial_kernel[grid_dw](
        go, x, rstd, dw_partial,
        N, D, N_SPLITS,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    dw = torch.empty((D,), dtype=torch.float32, device=device)
    REDUCE_BLOCK_D = 256
    _dw_reduce_kernel[(triton.cdiv(D, REDUCE_BLOCK_D),)](
        dw_partial, dw,
        D, N_SPLITS,
        BLOCK_D=REDUCE_BLOCK_D,
    )

    return (y, y, add_out, dw)