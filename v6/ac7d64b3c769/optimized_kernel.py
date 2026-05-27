import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rms_fwd_bwd_kernel(
    mm223_ptr, add62_ptr, mm226_ptr, weight_ptr,
    grad_input_ptr, grad_weight_partial_ptr,
    M, N: tl.constexpr, eps,
    BLOCK_N: tl.constexpr, NUM_M_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    a = tl.load(add62_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm223_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x_full = a + b
    x = (x_full).to(tl.bfloat16).to(tl.float32)

    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    go = tl.load(mm226_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    g = go * w
    gx = g * x
    sum_gx = tl.sum(gx, axis=0)

    coeff = rstd * rstd * sum_gx / N
    grad_in = rstd * (g - x * coeff)

    tl.store(grad_input_ptr + row * N + cols, grad_in.to(tl.bfloat16), mask=mask)

    gw_contrib = go * x * rstd
    m_block = row % NUM_M_BLOCKS
    rows_per_block = (M + NUM_M_BLOCKS - 1) // NUM_M_BLOCKS
    row_in_block = row // NUM_M_BLOCKS
    out_off = m_block * rows_per_block * N + row_in_block * N
    tl.store(grad_weight_partial_ptr + out_off + cols, gw_contrib, mask=mask)


@triton.jit
def _reduce_partial_kernel(
    partial_ptr, out_ptr, M_per_block, N, NUM_M_BLOCKS,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    base = pid_m * M_per_block * N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m_start in range(0, M_per_block, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        mask_m = rows < M_per_block
        ptrs = partial_ptr + base + rows[:, None] * N + cols[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + pid_m * N + cols, acc, mask=mask_n)


@triton.jit
def _final_reduce_kernel(
    inter_ptr, out_ptr, NUM_M_BLOCKS: tl.constexpr, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.static_range(NUM_M_BLOCKS):
        vals = tl.load(inter_ptr + i * N + cols, mask=mask, other=0.0)
        acc += vals
    tl.store(out_ptr + cols, acc, mask=mask)


def kernel_function(mm_223, add_62_recomputed, mm_226, wait_tensor_970):
    mm_223_c = mm_223.contiguous()
    add62_c = add_62_recomputed.contiguous()
    mm_226_c = mm_226.contiguous()
    weight_c = wait_tensor_970.contiguous()

    M = 8192
    N = 4096
    eps = 1e-5

    grad_input = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_223.device)

    NUM_M_BLOCKS = 32
    rows_per_block = (M + NUM_M_BLOCKS - 1) // NUM_M_BLOCKS
    grad_weight_partial = torch.empty((NUM_M_BLOCKS, rows_per_block, N), dtype=torch.float32, device=mm_223.device)

    BLOCK_N = 4096
    grid = (M,)
    _fused_add_rms_fwd_bwd_kernel[grid](
        mm_223_c, add62_c, mm_226_c, weight_c,
        grad_input, grad_weight_partial,
        M, N, eps,
        BLOCK_N=BLOCK_N,
        NUM_M_BLOCKS=NUM_M_BLOCKS,
        num_warps=8,
    )

    BLOCK_N2 = 64
    BLOCK_M2 = 32
    intermediate = torch.empty((NUM_M_BLOCKS, N), dtype=torch.float32, device=mm_223.device)
    grid2 = (triton.cdiv(N, BLOCK_N2), NUM_M_BLOCKS)
    _reduce_partial_kernel[grid2](
        grad_weight_partial, intermediate,
        rows_per_block, N, NUM_M_BLOCKS,
        BLOCK_N=BLOCK_N2, BLOCK_M=BLOCK_M2,
        num_warps=2,
    )

    grad_weight_fp32 = torch.empty((N,), dtype=torch.float32, device=mm_223.device)
    BLOCK_N3 = 128
    grid3 = (triton.cdiv(N, BLOCK_N3),)
    _final_reduce_kernel[grid3](
        intermediate, grad_weight_fp32,
        NUM_M_BLOCKS, N,
        BLOCK_N=BLOCK_N3,
        num_warps=4,
    )

    return (grad_input, grad_weight_fp32)