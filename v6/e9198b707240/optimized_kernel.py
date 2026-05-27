import triton
import triton.language as tl
import torch


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    y = x * rstd * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_dx_kernel(
    x_ptr, w_ptr, rstd_ptr,
    mm230_ptr, mm232_ptr, gi420_ptr,
    out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    dy1 = tl.load(mm230_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    dy2 = tl.load(mm232_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    dy = dy1 + dy2

    xhat = x * rstd
    dyw = dy * w

    mean_val = tl.sum(dyw * xhat, axis=0) / N

    dx = rstd * (dyw - xhat * mean_val)

    add_prev = tl.load(gi420_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    out = add_prev + dx
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dw_kernel_2d(
    x_ptr, rstd_ptr,
    mm230_ptr, mm232_ptr,
    dw_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_mask = col_offsets < N
    row_mask = row_offsets < M

    # Load rstd for the rows
    rstd = tl.load(rstd_ptr + row_offsets, mask=row_mask, other=0.0)  # [BLOCK_M]

    # 2D loads: [BLOCK_M, BLOCK_N]
    offs = row_offsets[:, None] * N + col_offsets[None, :]
    mask2d = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
    dy1 = tl.load(mm230_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
    dy2 = tl.load(mm232_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)

    dy = dy1 + dy2
    contrib = dy * x * rstd[:, None]
    acc = tl.sum(contrib, axis=0)

    tl.atomic_add(dw_ptr + col_offsets, acc, mask=col_mask)


def kernel_function(add_62_recomputed, _unsafe_view_986, mm_230, mm_232, getitem_420):
    x = add_62_recomputed.contiguous()
    w = _unsafe_view_986.contiguous()
    g1 = mm_230.contiguous()
    g2 = mm_232.contiguous()
    gprev = getitem_420.contiguous()

    M = 8192
    N = 4096

    x_2d = x.view(M, N)
    gprev_2d = gprev.view(M, N)

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 4096
    _rmsnorm_fwd_kernel[(M,)](
        x_2d, w, y, rstd,
        M, N, 1e-5,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    out_3d = torch.empty((1, M, N), dtype=torch.bfloat16, device=x.device)
    out_2d = out_3d.view(M, N)

    _rmsnorm_bwd_dx_kernel[(M,)](
        x_2d, w, rstd,
        g1, g2, gprev_2d,
        out_2d,
        M, N,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    dw = torch.zeros((N,), dtype=torch.float32, device=x.device)
    BLOCK_M_DW = 32
    BLOCK_N_DW = 128
    grid_dw = (triton.cdiv(N, BLOCK_N_DW), triton.cdiv(M, BLOCK_M_DW))
    _dw_kernel_2d[grid_dw](
        x_2d, rstd,
        g1, g2,
        dw,
        M, N,
        BLOCK_M=BLOCK_M_DW, BLOCK_N=BLOCK_N_DW,
        num_warps=4,
        num_stages=3,
    )

    return (y, y, out_3d, dw)