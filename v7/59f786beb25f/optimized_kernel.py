import torch
import triton
import triton.language as tl


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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    x_f = x.to(tl.float32)
    w_f = w.to(tl.float32)

    sum_sq = tl.sum(x_f * x_f, axis=0)
    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    tl.store(rstd_ptr + row, rstd)

    y = x_f * rstd * w_f
    tl.store(out_row_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(embedding, weight, eps: float = 1e-5):
    assert embedding.is_cuda and weight.is_cuda
    assert embedding.is_contiguous()
    N = weight.shape[0]
    assert embedding.shape[-1] == N

    orig_shape = embedding.shape
    x_flat = embedding.reshape(-1, N)
    M = x_flat.shape[0]

    out = torch.empty_like(x_flat)
    rstd_shape = list(orig_shape[:-1]) + [1]
    rstd = torch.empty(rstd_shape, dtype=torch.float32, device=embedding.device)
    rstd_flat = rstd.reshape(-1)

    BLOCK_SIZE = triton.next_power_of_2(N)

    if BLOCK_SIZE >= 4096:
        num_warps = 8
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    else:
        num_warps = 4

    grid = (M,)
    _rms_norm_kernel[grid](
        x_flat, weight, out, rstd_flat,
        x_flat.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=3,
    )

    out = out.reshape(orig_shape)
    return out, rstd