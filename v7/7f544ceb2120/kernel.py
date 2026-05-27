import torch
import triton
import triton.language as tl


@triton.jit
def _rms_bwd_kernel(
    mm_ptr, view1776_ptr, x_ptr, rstd_ptr, w_ptr, add218_ptr,
    out_add_ptr, dwbuf_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_off = row * N

    mm = tl.load(mm_ptr + row_off + cols, mask=mask, other=0.0)
    v = tl.load(view1776_ptr + row_off + cols, mask=mask, other=0.0)
    # Match PyTorch: add in bf16 (rounds intermediate result)
    dy_bf16 = (mm.to(tl.float32) + v.to(tl.float32)).to(tl.bfloat16)
    dy = dy_bf16.to(tl.float32)

    x_bf16 = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
    x = x_bf16.to(tl.float32)
    w_bf16 = tl.load(w_ptr + cols, mask=mask, other=0.0)
    w = w_bf16.to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    dyw = dy * w
    c = tl.sum(dyw * x, axis=0)
    mean_c = c / N

    dx = rstd * (dyw - x * rstd * rstd * mean_c)

    add218 = tl.load(add218_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    out = add218 + dx
    tl.store(out_add_ptr + row_off + cols, out.to(tl.bfloat16), mask=mask)

    dw_contrib = dy * x * rstd
    tl.store(dwbuf_ptr + row_off + cols, dw_contrib, mask=mask)


@triton.jit
def _dw_reduce_kernel(
    dwbuf_ptr, dw_out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_chunks = (M + BLOCK_M - 1) // BLOCK_M
    for i in range(0, num_chunks):
        r = i * BLOCK_M + rows
        m = r < M
        v = tl.load(dwbuf_ptr + r * N + col, mask=m, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(dw_out_ptr + col, s)


def kernel_function(mm_666, view_1776, add_recomputed, getitem_12_recomputed,
                    _unsafe_view_428, add_218):
    assert mm_666.is_cuda
    M = 8192
    N = 4096

    out_add = torch.empty_like(add_218)
    dwbuf = torch.empty((M, N), dtype=torch.float32, device=mm_666.device)
    dw_out = torch.empty((N,), dtype=torch.float32, device=mm_666.device)

    BLOCK_N = 4096
    grid = (M,)
    _rms_bwd_kernel[grid](
        mm_666, view_1776, add_recomputed, getitem_12_recomputed,
        _unsafe_view_428, add_218,
        out_add, dwbuf,
        M, N,
        BLOCK_N=BLOCK_N,
    )

    BLOCK_M = 256
    grid2 = (N,)
    _dw_reduce_kernel[grid2](
        dwbuf, dw_out,
        M, N,
        BLOCK_M=BLOCK_M,
    )

    return (out_add, dw_out)