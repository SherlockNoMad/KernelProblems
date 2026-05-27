import torch
import triton
import triton.language as tl


# Fused RMSNorm forward (compute rstd) + backward (grad_input, grad_weight partial).
# RMSNorm: y = x * rstd * w, where rstd = 1 / sqrt(mean(x^2) + eps)
# Backward:
#   grad_input = rstd * (grad_out * w - x * rstd^2 * mean(grad_out * w * x))
#   grad_weight = sum_over_rows(grad_out * x * rstd)
#
# We fuse:
#  Kernel 1 (per-row): compute rstd, then grad_input, and partial grad_weight accumulated per row.
#  Kernel 2: reduce partial grad_weight across rows.


@triton.jit
def _rms_fwd_bwd_kernel(
    x_ptr, w_ptr, dy_ptr,
    dx_ptr, dw_partial_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    dy_row = dy_ptr + row * N
    dx_row = dx_ptr + row * N
    dwp_row = dw_partial_ptr + row * N

    # Pass 1: compute sum(x^2) and sum(dy * w * x) for mean term
    sum_sq = tl.zeros((), dtype=tl.float32)
    sum_dyw_x = tl.zeros((), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        sum_dyw_x += tl.sum(dy * w * x, axis=0)

    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    # mean(dy*w*x) used in grad term, factor 1/N
    mean_dyw_x = sum_dyw_x / N

    # Pass 2: compute grad_input and partial grad_weight
    rstd3 = rstd * rstd * rstd
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)

        # grad_input = rstd * dy * w  -  x * rstd^3 * mean(dy*w*x)
        dx = rstd * dy * w - x * rstd3 * mean_dyw_x
        tl.store(dx_row + cols, dx.to(dx_row.dtype.element_ty), mask=mask)

        # partial grad_weight contribution: dy * x * rstd
        dwp = dy * x * rstd
        tl.store(dwp_row + cols, dwp.to(dwp_row.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_dw_kernel(
    dw_partial_ptr, dw_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    if col < N:
        acc = tl.zeros((), dtype=tl.float32)
        for off in range(0, M, BLOCK_M):
            rows = off + tl.arange(0, BLOCK_M)
            mask = rows < M
            vals = tl.load(dw_partial_ptr + rows * N + col, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(vals, axis=0)
        tl.store(dw_ptr + col, acc.to(dw_ptr.dtype.element_ty))


def kernel_function(x, weight, grad_out, eps=1e-5):
    """
    Fused RMSNorm forward + backward.
    
    Inputs:
        x: [..., N] input tensor
        weight: [N] weight tensor
        grad_out: [..., N] gradient w.r.t. output
    
    Returns:
        (grad_input, grad_weight)
    """
    assert x.is_cuda and weight.is_cuda and grad_out.is_cuda
    assert x.shape == grad_out.shape
    N = weight.shape[0]
    assert x.shape[-1] == N

    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, N)
    dy_2d = grad_out.contiguous().view(-1, N)
    M = x_2d.shape[0]

    grad_input = torch.empty_like(x_2d)
    # store partial dw in fp32 to reduce accumulation error
    dw_partial = torch.empty((M, N), dtype=torch.float32, device=x.device)
    grad_weight = torch.empty(N, dtype=weight.dtype, device=weight.device)

    BLOCK_N = 1024
    grid1 = (M,)
    _rms_fwd_bwd_kernel[grid1](
        x_2d, weight, dy_2d,
        grad_input, dw_partial,
        M, N,
        eps,
        BLOCK_N=BLOCK_N,
    )

    BLOCK_M = 256
    grid2 = (N,)
    _reduce_dw_kernel[grid2](
        dw_partial, grad_weight,
        M, N,
        BLOCK_M=BLOCK_M,
    )

    grad_input = grad_input.view(orig_shape)
    return grad_input, grad_weight