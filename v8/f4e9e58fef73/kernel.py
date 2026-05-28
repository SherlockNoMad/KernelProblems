import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_norm_kernel(
    x_ptr,       # input  [M, N]
    w_ptr,       # weight [N]
    y_ptr,       # output [M, N]
    rstd_ptr,    # rstd   [M]
    M,
    N: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMS norm:
      rstd = 1 / sqrt(mean(x^2) + eps)
      y = x * rstd * w
    One program per row.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    x_row_ptr = x_ptr + row * N
    y_row_ptr = y_ptr + row * N

    # Accumulate sum of squares in fp32
    sumsq = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean_sq = sumsq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    tl.store(rstd_ptr + row, rstd)

    # Apply normalization with weight
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_row_ptr + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def kernel_function(add_recomputed, weight):
    """
    Fused RMS Norm wrapper.

    Fused stages (all inside the Triton kernel):
      1. Reduction:  sumsq = sum(x^2) over last dim
      2. Compute:    rstd  = 1 / sqrt(sumsq/N + eps)
      3. Normalize:  y     = x * rstd * weight

    Returns:
      (view0, view1, rstd) where view0 == view1 are the normalized output
      reshaped to [8192, 4096], and rstd has shape [1, M, 1].
    """
    assert add_recomputed.is_cuda and weight.is_cuda
    assert add_recomputed.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16

    orig_shape = add_recomputed.shape
    N = weight.shape[0]
    assert orig_shape[-1] == N

    x = add_recomputed.contiguous()
    x2d = x.view(-1, N)
    M = x2d.shape[0]

    y2d = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # rstd shape mirrors aten._fused_rms_norm: leading dims of input + 1
    rstd_shape = list(orig_shape[:-1]) + [1]
    rstd = torch.empty(rstd_shape, device=x.device, dtype=torch.float32)

    # Pick BLOCK_SIZE
    BLOCK_SIZE = 1024
    if N < BLOCK_SIZE:
        # round up to next power of 2
        BLOCK_SIZE = 1
        while BLOCK_SIZE < N:
            BLOCK_SIZE *= 2

    grid = (M,)
    _fused_rms_norm_kernel[grid](
        x2d, weight, y2d, rstd,
        M, N, float(1e-5),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    view0 = y2d.view(M, N)
    view1 = y2d.view(M, N)
    return (view0, view1, rstd)