import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_bwd_dx_fused_kernel(
    a_ptr, b_ptr, c_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    add_extra_ptr,
    grad_in_ptr,
    grad_out_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * N + cols

    a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + off, mask=mask, other=0.0).to(tl.float32)
    # Mimic reference: add_tensor = view_3 + view_4 (bf16), then add view_5 (bf16)
    # The intermediate add_tensor is materialized in bf16.
    ab_bf16 = (a + b).to(grad_out_ptr.dtype.element_ty)
    ab_f32 = ab_bf16.to(tl.float32)
    g_bf16 = (ab_f32 + c).to(grad_out_ptr.dtype.element_ty)
    g = g_bf16.to(tl.float32)
    # store fused grad_out (bf16) for the dw pass
    tl.store(grad_out_ptr + off, g_bf16, mask=mask)

    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    gw = g * w
    x_hat = x * rstd
    dot = tl.sum(gw * x_hat, axis=0) / N
    dx = rstd * (gw - x_hat * dot)

    # Reference: dx is materialized as bf16 (getitem_2), then add_215 (bf16) + getitem_2 (bf16).
    dx_bf16 = dx.to(grad_in_ptr.dtype.element_ty)
    extra = tl.load(add_extra_ptr + off, mask=mask, other=0.0)  # keep bf16
    out_bf16 = extra + dx_bf16
    tl.store(grad_in_ptr + off, out_bf16, mask=mask)


@triton.jit
def _rms_bwd_dw_partial_kernel(
    grad_out_ptr,
    x_ptr,
    rstd_ptr,
    partial_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rmask = rows < M
    cmask = cols < N

    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0).to(tl.float32)

    offs = rows[:, None] * N + cols[None, :]
    mask2d = rmask[:, None] & cmask[None, :]

    x = tl.load(x_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
    g = tl.load(grad_out_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)

    contrib = g * x * rstd[:, None]
    partial = tl.sum(contrib, axis=0)

    tl.store(partial_ptr + pid_m * N + cols, partial, mask=cmask)


@triton.jit
def _dw_reduce_kernel(
    partial_ptr,
    grad_w_ptr,
    NUM_BLOCKS, N,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in range(0, NUM_BLOCKS, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        kmask = ks < NUM_BLOCKS
        offs = ks[:, None] * N + cols[None, :]
        mask2d = kmask[:, None] & cmask[None, :]
        v = tl.load(partial_ptr + offs, mask=mask2d, other=0.0)
        acc += tl.sum(v, axis=0)

    tl.store(grad_w_ptr + cols, acc, mask=cmask)


def kernel_function(add_1, _unsafe_view_450, mm_656, mm_658, mm_660, add_215):
    B, S, N = add_1.shape
    M = B * S
    eps = 1e-5

    x_2d = add_1.view(M, N)

    y = torch.empty((M, N), device=add_1.device, dtype=torch.bfloat16)
    rstd = torch.empty((M,), device=add_1.device, dtype=torch.float32)

    BLOCK_N_FULL = triton.next_power_of_2(N)
    _rms_fwd_kernel[(M,)](
        x_2d, _unsafe_view_450, y, rstd,
        M, N, eps,
        BLOCK_N=BLOCK_N_FULL,
        num_warps=8,
    )

    grad_out = torch.empty((M, N), device=add_1.device, dtype=torch.bfloat16)
    grad_in = torch.empty_like(add_1)

    _rms_bwd_dx_fused_kernel[(M,)](
        mm_656, mm_658, mm_660,
        x_2d, _unsafe_view_450, rstd,
        add_215.view(M, N),
        grad_in.view(M, N),
        grad_out,
        M, N,
        BLOCK_N=BLOCK_N_FULL,
        num_warps=8,
    )

    BLOCK_M = 64
    BLOCK_N_DW = 128
    NUM_ROW_BLOCKS = triton.cdiv(M, BLOCK_M)
    NUM_COL_BLOCKS = triton.cdiv(N, BLOCK_N_DW)

    partial = torch.empty((NUM_ROW_BLOCKS, N), device=add_1.device, dtype=torch.float32)
    _rms_bwd_dw_partial_kernel[(NUM_ROW_BLOCKS, NUM_COL_BLOCKS)](
        grad_out, x_2d, rstd, partial,
        M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N_DW,
        num_warps=4,
    )

    grad_w_fp32 = torch.empty((N,), device=add_1.device, dtype=torch.float32)
    BLOCK_N_RED = 128
    BLOCK_K = 16
    grid_red = (triton.cdiv(N, BLOCK_N_RED),)
    _dw_reduce_kernel[grid_red](
        partial, grad_w_fp32,
        NUM_ROW_BLOCKS, N,
        BLOCK_N=BLOCK_N_RED,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    return (y, y, y, grad_in, grad_w_fp32)