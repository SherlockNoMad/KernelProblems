import torch
import triton
import triton.language as tl


@triton.jit
def _rms_bwd_kernel(
    mm_ptr, view1776_ptr, x_ptr, rstd_ptr, w_ptr, add218_ptr,
    out_add_ptr, dw_partial_ptr,
    M, N, NUM_ROW_BLOCKS,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid_row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    w_bf16 = tl.load(w_ptr + cols, mask=mask, other=0.0)
    w = w_bf16.to(tl.float32)

    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    row_start = pid_row * ROWS_PER_PROG
    for i in range(0, ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            row_off = row * N

            mm = tl.load(mm_ptr + row_off + cols, mask=mask, other=0.0)
            v = tl.load(view1776_ptr + row_off + cols, mask=mask, other=0.0)
            dy_bf16 = (mm.to(tl.float32) + v.to(tl.float32)).to(tl.bfloat16)
            dy = dy_bf16.to(tl.float32)

            x_bf16 = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
            x = x_bf16.to(tl.float32)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)

            dyw = dy * w
            c = tl.sum(dyw * x, axis=0)
            mean_c = c / N

            dx = rstd * (dyw - x * rstd * rstd * mean_c)

            add218 = tl.load(add218_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
            out = add218 + dx
            tl.store(out_add_ptr + row_off + cols, out.to(tl.bfloat16), mask=mask)

            dw_acc += dy * x * rstd

    tl.store(dw_partial_ptr + pid_row * N + cols, dw_acc, mask=mask)


@triton.jit
def _dw_reduce_kernel(
    dw_partial_ptr, dw_out_ptr,
    NUM_ROW_BLOCKS, N,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, NUM_ROW_BLOCKS):
        v = tl.load(dw_partial_ptr + r * N + cols, mask=mask, other=0.0)
        acc += v
    tl.store(dw_out_ptr + cols, acc, mask=mask)


def kernel_function(mm_666, view_1776, add_recomputed, getitem_12_recomputed,
                    _unsafe_view_428, add_218):
    assert mm_666.is_cuda
    M = 8192
    N = 4096

    out_add = torch.empty_like(add_218)

    ROWS_PER_PROG = 16
    NUM_ROW_BLOCKS = (M + ROWS_PER_PROG - 1) // ROWS_PER_PROG

    dw_partial = torch.empty((NUM_ROW_BLOCKS, N), dtype=torch.float32, device=mm_666.device)
    dw_out = torch.empty((N,), dtype=torch.float32, device=mm_666.device)

    BLOCK_N = 4096
    grid = (NUM_ROW_BLOCKS,)
    _rms_bwd_kernel[grid](
        mm_666, view_1776, add_recomputed, getitem_12_recomputed,
        _unsafe_view_428, add_218,
        out_add, dw_partial,
        M, N, NUM_ROW_BLOCKS,
        BLOCK_N=BLOCK_N,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
    )

    BLOCK_N2 = 256
    grid2 = ((N + BLOCK_N2 - 1) // BLOCK_N2,)
    _dw_reduce_kernel[grid2](
        dw_partial, dw_out,
        NUM_ROW_BLOCKS, N,
        BLOCK_N=BLOCK_N2,
        BLOCK_R=1,
        num_warps=4,
    )

    return (out_add, dw_out)