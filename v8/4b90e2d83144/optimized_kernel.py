import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMS Norm: compute mean(x^2), rsqrt, multiply by weight - all in one pass."""
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * n_cols
    out_row_ptr = out_ptr + row * n_cols

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_row_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(embedding, weight, eps=1e-5):
    """
    Fused RMS norm kernel.
    Fuses: x^2 reduction, mean, rsqrt, scaling by weight - all in a single Triton kernel pass.
    Input: embedding [1, 8192, 4096] bf16, weight [4096] bf16
    Returns 3 identical views of the normalized output as [8192, 4096].
    """
    assert embedding.is_cuda and weight.is_cuda
    orig_shape = embedding.shape
    n_cols = orig_shape[-1]
    x_2d = embedding.reshape(-1, n_cols)
    n_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8 if BLOCK_SIZE >= 4096 else 4

    grid = (n_rows,)
    _rms_norm_kernel[grid](
        x_2d, weight, out,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    out_view = out.view(n_rows, n_cols)
    return (out_view, out_view, out_view)