import torch
import triton
import triton.language as tl

# Fused RMS Norm kernel
# Fuses: square, mean reduction, rsqrt, normalize, scale by weight
# All operations happen in a single Triton kernel pass per row.

@triton.jit
def _rms_norm_kernel(
    x_ptr, w_ptr, out_ptr, rstd_ptr,
    stride_row,
    N: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * stride_row
    out_row_ptr = out_ptr + row * stride_row

    # Compute sum of squares in fp32
    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Store rstd
    tl.store(rstd_ptr + row, rstd)

    # Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(out_row_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(embedding, weight, eps: float = 1e-5):
    """
    Fused RMS Norm: computes y = x * rsqrt(mean(x^2) + eps) * weight
    along the last dimension.

    Fusion: square -> mean reduction -> rsqrt -> normalize -> weight scaling,
    all done in a single Triton kernel per row.

    Returns (output_bf16, rstd_fp32) matching torch.ops.aten._fused_rms_norm.default.
    """
    assert embedding.is_cuda and weight.is_cuda
    assert embedding.is_contiguous()
    N = weight.shape[0]
    assert embedding.shape[-1] == N

    orig_shape = embedding.shape
    # Flatten all leading dims into rows
    x_flat = embedding.reshape(-1, N)
    M = x_flat.shape[0]

    out = torch.empty_like(x_flat)
    # rstd shape: leading dims with last dim = 1
    rstd_shape = list(orig_shape[:-1]) + [1]
    rstd = torch.empty(rstd_shape, dtype=torch.float32, device=embedding.device)
    rstd_flat = rstd.reshape(-1)

    # Choose BLOCK_SIZE
    BLOCK_SIZE = 1024
    if N < BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (M,)
    _rms_norm_kernel[grid](
        x_flat, weight, out, rstd_flat,
        x_flat.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    out = out.reshape(orig_shape)
    return out, rstd