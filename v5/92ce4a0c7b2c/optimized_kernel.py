import torch
import triton
import triton.language as tl


@triton.jit
def _fused_fwd_bwd_kernel(
    x_ptr, w_ptr,
    mm1_ptr, mm2_ptr, mm3_ptr,
    add215_ptr,
    y_ptr,
    out_grad_x_ptr, gyx_ptr,
    eps,
    N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = x * rstd

    y = xhat * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)

    gy1 = tl.load(mm1_ptr + row * N + cols, mask=mask, other=0.0)
    gy2 = tl.load(mm2_ptr + row * N + cols, mask=mask, other=0.0)
    gy3 = tl.load(mm3_ptr + row * N + cols, mask=mask, other=0.0)
    gy12 = (gy1.to(tl.float32) + gy2.to(tl.float32)).to(tl.bfloat16)
    gy_bf16 = (gy12.to(tl.float32) + gy3.to(tl.float32)).to(tl.bfloat16)
    gy = gy_bf16.to(tl.float32)

    c1 = gy * w
    mean_term = tl.sum(c1 * xhat, axis=0) / N
    grad_x = rstd * (c1 - xhat * mean_term)

    add215 = tl.load(add215_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    out = add215 + grad_x
    tl.store(out_grad_x_ptr + row * N + cols, out.to(out_grad_x_ptr.dtype.element_ty), mask=mask)

    gyx = gy * xhat
    tl.store(gyx_ptr + row * N + cols, gyx, mask=mask)


@triton.jit
def _grad_w_split_kernel(
    gyx_ptr, partial_ptr,
    M, N: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    mask_c = cols < N

    rows_per_block = M // NUM_M_BLOCKS
    m_begin = pid_m * rows_per_block
    m_end = m_begin + rows_per_block

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m_start in range(m_begin, m_end, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        ptrs = gyx_ptr + rows[:, None] * N + cols[None, :]
        vals = tl.load(ptrs, mask=mask_c[None, :], other=0.0)
        acc += tl.sum(vals, axis=0)

    tl.store(partial_ptr + pid_m * N + cols, acc, mask=mask_c)


@triton.jit
def _grad_w_final_kernel(
    partial_ptr, gw_ptr,
    N: tl.constexpr, NUM_M_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(NUM_M_BLOCKS):
        v = tl.load(partial_ptr + i * N + cols, mask=mask, other=0.0)
        acc += v
    tl.store(gw_ptr + cols, acc, mask=mask)


def kernel_function(getitem_818, add_1, mm_656, mm_658, mm_660, add_215):
    assert add_1.shape == (1, 8192, 4096)

    weight = getitem_818.contiguous().view(-1)

    M = 8192
    N = 4096
    eps = 1e-5

    x_flat = add_1.view(M, N)
    add215_flat = add_215.view(M, N)

    y = torch.empty((M, N), dtype=torch.bfloat16, device=add_1.device)
    out_grad_x = torch.empty((1, M, N), dtype=torch.bfloat16, device=add_1.device)
    gyx_temp = torch.empty((M, N), dtype=torch.float32, device=add_1.device)

    BLOCK_N = 4096
    _fused_fwd_bwd_kernel[(M,)](
        x_flat, weight,
        mm_656, mm_658, mm_660,
        add215_flat,
        y,
        out_grad_x.view(M, N), gyx_temp,
        eps,
        N=N, BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    grad_w = torch.empty((N,), dtype=torch.float32, device=add_1.device)

    BLOCK_N_R = 128
    NUM_M_BLOCKS = 32
    BLOCK_M = 32
    grid_partial = (triton.cdiv(N, BLOCK_N_R), NUM_M_BLOCKS)
    partial = torch.empty((NUM_M_BLOCKS, N), dtype=torch.float32, device=add_1.device)
    _grad_w_split_kernel[grid_partial](
        gyx_temp, partial,
        M, N=N, NUM_M_BLOCKS=NUM_M_BLOCKS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_R,
        num_warps=4,
    )

    BLOCK_N_F = 128
    grid_final = (triton.cdiv(N, BLOCK_N_F),)
    _grad_w_final_kernel[grid_final](
        partial, grad_w,
        N=N, NUM_M_BLOCKS=NUM_M_BLOCKS, BLOCK_N=BLOCK_N_F,
        num_warps=2,
    )

    return (y, y, y, out_grad_x, grad_w)