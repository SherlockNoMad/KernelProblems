import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    mm664_ptr, mm666_ptr, x_ptr, rstd_ptr, w_ptr, add218_ptr,
    out_ptr, dw_partial_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            base = row * N
            mm664 = tl.load(mm664_ptr + base + offs, mask=mask, other=0.0)
            mm666 = tl.load(mm666_ptr + base + offs, mask=mask, other=0.0)
            dy = (mm664 + mm666).to(tl.float32)

            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)
            add218 = tl.load(add218_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            x_hat = x * rstd
            dy_hat = dy * w
            c = tl.sum(dy_hat * x_hat, axis=0) / N
            dx = rstd * (dy_hat - x_hat * c)

            out = add218 + dx
            tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)

            dw_acc += dy * x_hat

    tl.store(dw_partial_ptr + pid * N + offs, dw_acc, mask=mask)


@triton.jit
def _dw_reduce_kernel(
    dw_partial_ptr, out_dw_ptr,
    N, NUM_PARTIALS,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(NUM_PARTIALS):
        v = tl.load(dw_partial_ptr + i * N + offs, mask=mask, other=0.0)
        acc += v
    tl.store(out_dw_ptr + offs, acc, mask=mask)


def kernel_function(
    mm_664,
    mm_666,
    add_recomputed,
    getitem_12_recomputed,
    _unsafe_view_428,
    add_218,
):
    mm_664_c = mm_664.contiguous()
    mm_666_c = mm_666.contiguous()
    x_c = add_recomputed.contiguous()
    rstd_c = getitem_12_recomputed.contiguous()
    w_c = _unsafe_view_428.contiguous()
    add218_c = add_218.contiguous()

    M = 8192
    N = 4096

    out_view = torch.empty((M, N), dtype=torch.bfloat16, device=mm_664.device)

    ROWS_PER_PROG = 16
    num_progs = M // ROWS_PER_PROG

    dw_partial = torch.empty((num_progs, N), dtype=torch.float32, device=mm_664.device)

    _fused_kernel[(num_progs,)](
        mm_664_c, mm_666_c, x_c, rstd_c, w_c, add218_c,
        out_view, dw_partial,
        M, N,
        BLOCK_SIZE=4096,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
        num_stages=3,
    )

    out_dw = torch.empty((N,), dtype=torch.float32, device=mm_664.device)
    BLOCK_N = 256
    _dw_reduce_kernel[(triton.cdiv(N, BLOCK_N),)](
        dw_partial, out_dw,
        N, num_progs,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return (out_view, out_dw)