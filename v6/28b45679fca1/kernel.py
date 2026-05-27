import torch
import triton
import triton.language as tl


# Fused stages:
#  Kernel A (_fwd_kernel): computes add_tensor = add_62_recomputed + view(mm_223) and per-row rstd
#                          for RMSNorm (1 / sqrt(mean(x^2) + eps)) in a single pass.
#  Kernel B (_bwd_dx_kernel): computes grad_input for the RMSNorm backward in a single pass per row,
#                             reusing add_tensor and rstd from Kernel A.
#  Kernel C (_bwd_dw_kernel): column-wise reduction across rows to compute grad_weight (fp32 out).


@triton.jit
def _fwd_kernel(
    mm_ptr, add_ptr, x_ptr, rstd_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * N + cols

    a = tl.load(mm_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(add_ptr + off, mask=mask, other=0.0).to(tl.float32)
    x = a + b
    # Store add_tensor in bf16 (matches reference dtype)
    tl.store(x_ptr + off, x.to(x_ptr.dtype.element_ty), mask=mask)

    sq = x * x
    var = tl.sum(sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _bwd_dx_kernel(
    go_ptr, x_ptr, w_ptr, rstd_ptr, gi_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * N + cols

    go = tl.load(go_ptr + off, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    gw_x = go * w
    # mean over feature dim of gw_x * x_hat
    s = tl.sum(gw_x * x_hat, axis=0) / N
    gi = rstd * (gw_x - x_hat * s)

    tl.store(gi_ptr + off, gi.to(gi_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_dw_kernel(
    go_ptr, x_ptr, rstd_ptr, gw_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        mask2d = row_mask[:, None] & col_mask[None, :]
        off2d = rows[:, None] * N + cols[None, :]

        x = tl.load(x_ptr + off2d, mask=mask2d, other=0.0).to(tl.float32)
        go = tl.load(go_ptr + off2d, mask=mask2d, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        x_hat = x * rstd[:, None]
        acc += tl.sum(go * x_hat, axis=0)

    tl.store(gw_ptr + cols, acc, mask=col_mask)


def kernel_function(mm_223, add_62_recomputed, mm_226, view_3253):
    """
    Fused RMSNorm forward+backward.

    Computes:
      add_tensor = add_62_recomputed + mm_223.view(1, 8192, 4096)
      rstd       = 1 / sqrt(mean(add_tensor^2, dim=-1) + eps)
      grad_input = rms_norm_backward(grad_out=mm_226.view(...), x=add_tensor, weight=view_3253, rstd)
      grad_weight= sum_over_batch(grad_out * (x * rstd))  (fp32)
    """
    assert mm_223.is_cuda and add_62_recomputed.is_cuda and mm_226.is_cuda and view_3253.is_cuda
    assert mm_223.shape == (8192, 4096)
    assert add_62_recomputed.shape == (1, 8192, 4096)
    assert mm_226.shape == (8192, 4096)
    assert view_3253.shape == (4096,)

    eps = 1e-5
    M = 8192
    N = 4096

    device = mm_223.device

    # Ensure contiguous for simple indexing
    mm_223_c = mm_223.contiguous()
    add_62_c = add_62_recomputed.contiguous()
    mm_226_c = mm_226.contiguous()
    w_c = view_3253.contiguous()

    add_tensor = torch.empty((1, M, N), dtype=mm_223.dtype, device=device)
    rstd = torch.empty((M,), dtype=torch.float32, device=device)

    BLOCK_N = 4096  # N fits in one block (power of 2)

    # Kernel A: build add_tensor and compute rstd
    _fwd_kernel[(M,)](
        mm_223_c, add_62_c, add_tensor, rstd,
        M, N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    grad_input = torch.empty((1, M, N), dtype=mm_223.dtype, device=device)

    # Kernel B: compute grad_input
    _bwd_dx_kernel[(M,)](
        mm_226_c, add_tensor, w_c, rstd, grad_input,
        M, N,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    grad_weight = torch.empty((N,), dtype=torch.float32, device=device)

    # Kernel C: weight gradient reduction
    BLOCK_M_C = 64
    BLOCK_N_C = 128
    grid_c = (triton.cdiv(N, BLOCK_N_C),)
    _bwd_dw_kernel[grid_c](
        mm_226_c, add_tensor, rstd, grad_weight,
        M, N,
        BLOCK_M=BLOCK_M_C,
        BLOCK_N=BLOCK_N_C,
        num_warps=4,
    )

    return (grad_input, grad_weight)