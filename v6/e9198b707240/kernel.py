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
    out_ptr,           # getitem_420 + dx, shape [M, N], bf16
    dw_partial_ptr,    # partial sums per row [M, N], fp32 - too much memory
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

    # mean(dyw * xhat) over N
    mean_val = tl.sum(dyw * xhat, axis=0) / N

    dx = rstd * (dyw - xhat * mean_val)

    add_prev = tl.load(gi420_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    out = add_prev + dx
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

    # store partial dw contribution = dy * xhat
    dw_contrib = dy * xhat
    tl.store(dw_partial_ptr + row * N + cols, dw_contrib, mask=mask)


@triton.jit
def _dw_reduce_kernel(
    dw_partial_ptr,  # [M, N] fp32
    dw_ptr,          # [N] fp32
    M, N,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        mask = rows < M
        vals = tl.load(dw_partial_ptr + rows * N + col, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(dw_ptr + col, acc)


def kernel_function(add_62_recomputed, _unsafe_view_986, mm_230, mm_232, getitem_420):
    """
    Fused RMSNorm forward + backward pipeline.
    
    Fused stages:
    - Kernel 1: forward RMSNorm computing y and rstd (replaces _fused_rms_norm)
    - Kernel 2: per-row backward: computes dy = mm_230 + mm_232, dx, and
                stores (getitem_420 + dx). Also writes per-row partial dw=dy*xhat.
    - Kernel 3: column-wise reduction over rows to produce dweight in fp32.
    """
    assert add_62_recomputed.is_cuda
    x = add_62_recomputed.contiguous()
    w = _unsafe_view_986.contiguous()
    g1 = mm_230.contiguous()
    g2 = mm_232.contiguous()
    gprev = getitem_420.contiguous()

    # Flatten leading dims
    # x: [1, 8192, 4096] -> [8192, 4096]
    M = 8192
    N = 4096
    assert w.numel() == N

    x_2d = x.view(M, N)
    gprev_2d = gprev.view(M, N)

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 4096  # N is exactly 4096
    grid_fwd = (M,)
    _rmsnorm_fwd_kernel[grid_fwd](
        x_2d, w, y, rstd,
        M, N, 1e-5,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    # Backward
    out_3d = torch.empty((1, M, N), dtype=torch.bfloat16, device=x.device)
    out_2d = out_3d.view(M, N)
    dw_partial = torch.empty((M, N), dtype=torch.float32, device=x.device)

    _rmsnorm_bwd_dx_kernel[grid_fwd](
        x_2d, w, rstd,
        g1, g2, gprev_2d,
        out_2d, dw_partial,
        M, N,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    dw = torch.empty((N,), dtype=torch.float32, device=x.device)
    grid_dw = (N,)
    _dw_reduce_kernel[grid_dw](
        dw_partial, dw,
        M, N,
        BLOCK_M=256,
        num_warps=4,
    )

    # Outputs: view_default, view_default_1 (both same y), add_tensor_1, dw_fp32
    return (y, y, out_3d, dw)