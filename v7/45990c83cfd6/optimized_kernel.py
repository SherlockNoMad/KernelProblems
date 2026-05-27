import torch
import triton
import triton.language as tl


@triton.jit
def _rms_compute_rstd_and_dx(
    x_ptr, w_ptr, dy_ptr,
    dx_ptr, rstd_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    dy_row = dy_ptr + row * N
    dx_row = dx_ptr + row * N

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    dyw = dy * w
    sum_dyw_x = tl.sum(dyw * x, axis=0)

    inv_N = 1.0 / N.to(tl.float32)
    mean_sq = sum_sq * inv_N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    mean_dyw_x = sum_dyw_x * inv_N
    rstd3 = rstd * rstd * rstd

    tl.store(rstd_ptr + row, rstd)

    dx = rstd * dyw - x * (rstd3 * mean_dyw_x)
    tl.store(dx_row + cols, dx.to(dx_row.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_dw_split_k(
    x_ptr, dy_ptr, rstd_ptr,
    dw_partial_ptr,
    M, N, SPLITS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    rows_per_split = tl.cdiv(M, SPLITS)
    row_start = pid_k * rows_per_split
    row_end = tl.minimum(row_start + rows_per_split, M)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    rows_off = tl.arange(0, BLOCK_M)

    for off in range(row_start, row_end, BLOCK_M):
        r = off + rows_off
        rmask = r < row_end
        offs = r[:, None] * N + cols[None, :]
        mask2d = rmask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + r, mask=rmask, other=0.0)
        acc += tl.sum(dy * x * rstd[:, None], axis=0)

    out_ptr = dw_partial_ptr + pid_k * N + cols
    tl.store(out_ptr, acc, mask=col_mask)


@triton.jit
def _final_reduce_dw(
    dw_partial_ptr, dw_ptr,
    N, SPLITS,
    BLOCK_N: tl.constexpr,
    SPLITS_C: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, SPLITS_C):
        v = tl.load(dw_partial_ptr + k * N + cols, mask=mask, other=0.0)
        acc += v
    tl.store(dw_ptr + cols, acc.to(dw_ptr.dtype.element_ty), mask=mask)


def kernel_function(x, weight, grad_out, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and grad_out.is_cuda
    N = weight.shape[0]
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, N)
    dy_2d = grad_out.contiguous().view(-1, N)
    M = x_2d.shape[0]

    grad_input = torch.empty_like(x_2d)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    grad_weight = torch.empty(N, dtype=weight.dtype, device=weight.device)

    BLOCK_N_FWD = triton.next_power_of_2(N)
    _rms_compute_rstd_and_dx[(M,)](
        x_2d, weight, dy_2d,
        grad_input, rstd,
        M, N,
        eps,
        BLOCK_N=BLOCK_N_FWD,
        num_warps=8,
        num_stages=3,
    )

    BLOCK_N = 64
    BLOCK_M = 32
    SPLITS = 16
    dw_partial = torch.empty((SPLITS, N), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(N, BLOCK_N), SPLITS)
    _reduce_dw_split_k[grid](
        x_2d, dy_2d, rstd,
        dw_partial,
        M, N, SPLITS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=2,
        num_stages=4,
    )

    BLOCK_N_FINAL = 256
    grid2 = (triton.cdiv(N, BLOCK_N_FINAL),)
    _final_reduce_dw[grid2](
        dw_partial, grad_weight,
        N, SPLITS,
        BLOCK_N=BLOCK_N_FINAL,
        SPLITS_C=SPLITS,
        num_warps=4,
    )

    grad_input = grad_input.view(orig_shape)
    return grad_input, grad_weight