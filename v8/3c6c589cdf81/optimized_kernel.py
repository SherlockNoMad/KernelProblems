import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rms_fwd_dx_kernel(
    mm220_ptr, add61_ptr, mm223_ptr, mm226_ptr, w_ptr,
    x_out_ptr, dx_ptr, partial_dw_ptr,
    M, N, eps,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    base = row * N
    a = tl.load(mm220_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(add61_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(mm223_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(mm226_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x = a + b + c

    tl.store(x_out_ptr + base + cols, x.to(tl.bfloat16), mask=mask)

    s = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(s + eps)

    dy_w = dy * w
    sum_dy_w_x = tl.sum(dy_w * x, axis=0)

    dx = rstd * (dy_w - x * (rstd * rstd) * sum_dy_w_x / N)
    tl.store(dx_ptr + base + cols, dx.to(tl.bfloat16), mask=mask)

    # Accumulate partial dw: dy * x * rstd
    # Bucket by row % NUM_M_BLOCKS
    bucket = row % NUM_M_BLOCKS
    contrib = dy * x * rstd
    tl.store(partial_dw_ptr + bucket * N + cols, contrib, mask=mask)


@triton.jit
def _fused_add_rms_fwd_dx_kernel_v2(
    mm220_ptr, add61_ptr, mm223_ptr, mm226_ptr, w_ptr,
    x_out_ptr, dx_ptr, partial_dw_ptr,
    M, N, eps,
    NUM_M_BLOCKS: tl.constexpr,
    ROWS_PER_BUCKET: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bucket = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    acc_dw = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in range(ROWS_PER_BUCKET):
        row = bucket * ROWS_PER_BUCKET + i
        base = row * N
        a = tl.load(mm220_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(add61_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(mm223_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(mm226_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)

        x = a + b + c
        tl.store(x_out_ptr + base + cols, x.to(tl.bfloat16), mask=mask)

        s = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(s + eps)

        dy_w = dy * w
        sum_dy_w_x = tl.sum(dy_w * x, axis=0)

        dx = rstd * (dy_w - x * (rstd * rstd) * sum_dy_w_x / N)
        tl.store(dx_ptr + base + cols, dx.to(tl.bfloat16), mask=mask)

        acc_dw += dy * x * rstd

    tl.store(partial_dw_ptr + bucket * N + cols, acc_dw, mask=mask)


@triton.jit
def _dweight_reduce_kernel(
    partial_ptr, dw_ptr,
    N, NUM_M_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m in range(NUM_M_BLOCKS):
        vals = tl.load(partial_ptr + m * N + col_offsets, mask=col_mask, other=0.0)
        acc += vals

    tl.store(dw_ptr + col_offsets, acc, mask=col_mask)


def kernel_function(mm_220, add_61, mm_223, mm_226, view_3253):
    M = 8192
    N = 4096
    eps = 1e-5

    device = mm_220.device

    x_buf = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    dx_buf = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    dw_buf = torch.empty((N,), dtype=torch.float32, device=device)

    BLOCK_N = triton.next_power_of_2(N)

    # Fuse partial dw accumulation into the forward kernel
    # Use 256 buckets (rows_per_bucket=32) to saturate 132 SMs with ~2x oversubscription
    NUM_M_BLOCKS = 256
    ROWS_PER_BUCKET = M // NUM_M_BLOCKS  # 32

    partial = torch.empty((NUM_M_BLOCKS, N), dtype=torch.float32, device=device)

    _fused_add_rms_fwd_dx_kernel_v2[(NUM_M_BLOCKS,)](
        mm_220, add_61, mm_223, mm_226, view_3253,
        x_buf, dx_buf, partial,
        M, N, eps,
        NUM_M_BLOCKS=NUM_M_BLOCKS,
        ROWS_PER_BUCKET=ROWS_PER_BUCKET,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    BLOCK_N3 = 128
    grid2 = (triton.cdiv(N, BLOCK_N3),)
    _dweight_reduce_kernel[grid2](
        partial, dw_buf,
        N, NUM_M_BLOCKS=NUM_M_BLOCKS,
        BLOCK_N=BLOCK_N3,
        num_warps=4,
    )

    return dx_buf, dw_buf