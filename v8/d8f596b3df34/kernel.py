import torch
import triton
import triton.language as tl


@triton.jit
def _dx_kernel(
    mm664_ptr, mm666_ptr, x_ptr, rstd_ptr, w_ptr, add218_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    base = row * N
    mm664 = tl.load(mm664_ptr + base + offs, mask=mask, other=0.0)
    mm666 = tl.load(mm666_ptr + base + offs, mask=mask, other=0.0)
    # Match reference: add in bf16 first to produce bf16 dy
    dy_bf16 = (mm664 + mm666)
    dy = dy_bf16.to(tl.float32)

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    add218 = tl.load(add218_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # dy_hat = dy * w  (gradient through scaling)
    dy_hat = dy * w
    # c = mean(dy_hat * x_normalized) where x_normalized = x * rstd
    # dx = rstd * (dy_hat - x_normalized * c)
    x_hat = x * rstd
    c = tl.sum(dy_hat * x_hat, axis=0) / N
    dx = rstd * (dy_hat - x_hat * c)

    out = add218 + dx
    tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _dw_kernel(
    mm664_ptr, mm666_ptr, x_ptr, rstd_ptr, out_ptr,
    M, N,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """
    dw_i = sum_row( dy_row,i * x_row,i * rstd_row )
    Where dy is computed in bf16 first to match reference precision.
    """
    pid = tl.program_id(0)
    col_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offs < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        row_offs = m_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < M
        full_mask = row_mask[:, None] & col_mask[None, :]

        idx = row_offs[:, None] * N + col_offs[None, :]
        mm664 = tl.load(mm664_ptr + idx, mask=full_mask, other=0.0)
        mm666 = tl.load(mm666_ptr + idx, mask=full_mask, other=0.0)
        # Add in bf16 first to match reference (view + view + add produces bf16 dy)
        dy_bf16 = mm664 + mm666
        dy = dy_bf16.to(tl.float32)

        x = tl.load(x_ptr + idx, mask=full_mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

        # x_normalized = x * rstd (this is what the rms_norm forward output equals before w scaling)
        x_hat = x * rstd[:, None]
        contrib = dy * x_hat
        acc += tl.sum(contrib, axis=0)

    tl.store(out_ptr + col_offs, acc, mask=col_mask)


def kernel_function(
    mm_664,
    mm_666,
    add_recomputed,
    getitem_12_recomputed,
    _unsafe_view_428,
    add_218,
):
    assert mm_664.is_cuda and mm_666.is_cuda
    assert mm_664.shape == (8192, 4096)
    assert mm_666.shape == (8192, 4096)
    assert add_recomputed.shape == (1, 8192, 4096)
    assert getitem_12_recomputed.shape == (1, 8192, 1)
    assert _unsafe_view_428.shape == (4096,)
    assert add_218.shape == (1, 8192, 4096)

    mm_664_c = mm_664.contiguous()
    mm_666_c = mm_666.contiguous()
    x_c = add_recomputed.contiguous()
    rstd_c = getitem_12_recomputed.contiguous()
    w_c = _unsafe_view_428.contiguous()
    add218_c = add_218.contiguous()

    M = 8192
    N = 4096

    out_view = torch.empty((M, N), dtype=torch.bfloat16, device=mm_664.device)
    out_dw = torch.empty((N,), dtype=torch.float32, device=mm_664.device)

    BLOCK_SIZE = 4096
    grid_dx = (M,)
    _dx_kernel[grid_dx](
        mm_664_c, mm_666_c, x_c, rstd_c, w_c, add218_c, out_view,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    BLOCK_N = 64
    BLOCK_M = 128
    grid_dw = (triton.cdiv(N, BLOCK_N),)
    _dw_kernel[grid_dw](
        mm_664_c, mm_666_c, x_c, rstd_c, out_dw,
        M, N,
        BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    return (out_view, out_dw)