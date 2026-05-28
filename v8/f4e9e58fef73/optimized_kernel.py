import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    x_row_ptr = x_ptr + row * N
    y_row_ptr = y_ptr + row * N

    sumsq = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean_sq = sumsq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    tl.store(rstd_ptr + row, rstd)

    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        y = x * rstd * w
        tl.store(y_row_ptr + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def kernel_function(add_recomputed, weight):
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
    rstd_shape = list(orig_shape[:-1]) + [1]
    rstd = torch.empty(rstd_shape, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 2048
    if N < BLOCK_SIZE:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < N:
            BLOCK_SIZE *= 2

    grid = (M,)
    _fused_rms_norm_kernel[grid](
        x2d, weight, y2d, rstd,
        M, N, float(1e-5),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )

    view0 = y2d.view(M, N)
    view1 = y2d.view(M, N)
    return (view0, view1, rstd)