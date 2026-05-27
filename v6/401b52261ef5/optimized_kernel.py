import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = x * rstd * w
    tl.store(y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_dx_fused_kernel(
    a_ptr, b_ptr, x_ptr, w_ptr, rstd_ptr, prev_ptr, dx_out_ptr,
    N, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols
    a_bf = tl.load(a_ptr + off, mask=mask, other=0.0)
    b_bf = tl.load(b_ptr + off, mask=mask, other=0.0)
    dy_bf = a_bf + b_bf
    dy = dy_bf.to(tl.float32)
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    prev = tl.load(prev_ptr + off, mask=mask, other=0.0)

    x_hat = x * rstd
    wdy = w * dy
    mean = tl.sum(x_hat * wdy, axis=0) / N
    dx = (wdy - x_hat * mean) * rstd
    dx_bf16 = dx.to(tl.bfloat16)
    out = prev + dx_bf16
    tl.store(dx_out_ptr + off, out, mask=mask)


@triton.jit
def rmsnorm_bwd_dw_partial_kernel(
    a_ptr, b_ptr, x_ptr, rstd_ptr, partial_ptr,
    M, N, ROWS_PER_PROG: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    row_start = pid_m * ROWS_PER_PROG
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            offs = row * N + cols
            a_bf = tl.load(a_ptr + offs, mask=col_mask, other=0.0)
            b_bf = tl.load(b_ptr + offs, mask=col_mask, other=0.0)
            dy_bf = a_bf + b_bf
            dy = dy_bf.to(tl.float32)
            x = tl.load(x_ptr + offs, mask=col_mask, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)
            x_hat = x * rstd
            acc += dy * x_hat
    out_offs = pid_m * N + cols
    tl.store(partial_ptr + out_offs, acc, mask=col_mask)


@triton.jit
def reduce_partials_kernel(partial_ptr, dw_ptr, num_partials, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(num_partials):
        v = tl.load(partial_ptr + i * N + cols, mask=mask, other=0.0)
        acc += v
    tl.store(dw_ptr + cols, acc, mask=mask)


@triton.jit
def make_weight_kernel(src_ptr, dst_ptr, rows, cols_per_row, src_stride0, src_stride1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = rows * cols_per_row
    mask = offs < total
    r = offs // cols_per_row
    c = offs % cols_per_row
    src_idx = r * src_stride0 + c * src_stride1
    v = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, v, mask=mask)


def kernel_function(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420):
    device = add_62_recomputed.device
    N = 4096
    M = 8192

    weight = torch.empty((N,), dtype=torch.bfloat16, device=device)
    src_stride0 = getitem_1624.stride(0)
    src_stride1 = getitem_1624.stride(1)
    BLOCK_W = 1024
    grid_w = (triton.cdiv(N, BLOCK_W),)
    make_weight_kernel[grid_w](
        getitem_1624, weight, 8, 512, src_stride0, src_stride1, BLOCK=BLOCK_W
    )

    x_flat = add_62_recomputed.view(M, N)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    rstd = torch.empty((M,), dtype=torch.float32, device=device)

    BLOCK_N_FWD = triton.next_power_of_2(N)
    rmsnorm_fwd_kernel[(M,)](
        x_flat, weight, y, rstd, N, 1e-5, BLOCK=BLOCK_N_FWD,
        num_warps=8
    )

    view_default = y
    view_default_1 = y

    add_tensor_1 = torch.empty_like(getitem_420)
    prev_flat = getitem_420.view(M, N)
    out_flat = add_tensor_1.view(M, N)
    rmsnorm_bwd_dx_fused_kernel[(M,)](
        mm_230, mm_232, x_flat, weight, rstd, prev_flat, out_flat, N,
        BLOCK=BLOCK_N_FWD, num_warps=8
    )

    BLOCK_N_DW = 256
    ROWS_PER_PROG = 64
    num_m_blocks = triton.cdiv(M, ROWS_PER_PROG)
    num_n_blocks = triton.cdiv(N, BLOCK_N_DW)
    partials = torch.empty((num_m_blocks, N), dtype=torch.float32, device=device)
    grid_dw = (num_m_blocks, num_n_blocks)
    rmsnorm_bwd_dw_partial_kernel[grid_dw](
        mm_230, mm_232, x_flat, rstd, partials,
        M, N, ROWS_PER_PROG=ROWS_PER_PROG, BLOCK_N=BLOCK_N_DW,
        num_warps=4
    )

    dw = torch.empty((N,), dtype=torch.float32, device=device)
    BLOCK_RED = 256
    grid_red = (triton.cdiv(N, BLOCK_RED),)
    reduce_partials_kernel[grid_red](
        partials, dw, num_m_blocks, N, BLOCK_N=BLOCK_RED
    )

    return (view_default, view_default_1, add_tensor_1, dw)