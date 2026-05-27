import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    w_stride_row, w_stride_col,
    W_COLS: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    x_row_ptr = x_ptr + row * n_cols
    out_row_ptr = out_ptr + row * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / n_cols
    rstd = tl.rsqrt(mean_sq + eps)

    # Gather weight from strided [8, 512] layout directly
    w_row = cols // W_COLS
    w_col = cols % W_COLS
    w_offs = w_row * w_stride_row + w_col * w_stride_col
    w = tl.load(w_ptr + w_offs, mask=mask, other=0.0).to(tl.float32)

    y = x * rstd * w

    tl.store(out_row_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(getitem_1619, add_62):
    assert getitem_1619.dtype == torch.bfloat16
    assert add_62.dtype == torch.bfloat16

    # add_62 is already contiguous in last dim with stride [33554432, 4096, 1]
    # View as [8192, 4096] without copy
    n_cols = add_62.shape[-1]
    x = add_62.view(-1, n_cols)
    n_rows = x.shape[0]

    out = torch.empty_like(x)

    stride_row_w, stride_col_w = getitem_1619.stride()
    W_COLS = getitem_1619.shape[1]

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    grid = (n_rows,)
    _rms_norm_kernel[grid](
        x, getitem_1619, out,
        stride_row_w, stride_col_w,
        W_COLS,
        n_cols,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4,
    )

    out_view = out.view(8192, 4096)
    return (out_view, out_view)