import torch
import triton
import triton.language as tl


@triton.jit
def _gather_weight_kernel(src_ptr, dst_ptr, N, stride_row, stride_col, COLS: tl.constexpr, BLOCK: tl.constexpr):
    """Gather the strided [8, 512] weight tensor into a contiguous [4096] buffer."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    row = offs // COLS
    col = offs % COLS
    src_offs = row * stride_row + col * stride_col
    vals = tl.load(src_ptr + src_offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def _rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMS norm over the last dimension.
    Stages fused:
      1. Load row of x (bf16 -> fp32)
      2. Compute sum of squares -> mean -> rsqrt
      3. Multiply by weight
      4. Store output (bf16)
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return

    x_row_ptr = x_ptr + row * n_cols
    out_row_ptr = out_ptr + row * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_row_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(getitem_1619, add_62):
    """
    Fused RMS norm implementation.
    
    Fused stages:
      - Stage A: gather strided weight [8,512] -> contiguous [4096] (Triton kernel)
      - Stage B: RMS norm over last dim, multiply by weight, cast to bf16 (Triton kernel)
    
    The two outputs are views of the same tensor (as in the reference).
    """
    assert getitem_1619.dtype == torch.bfloat16
    assert add_62.dtype == torch.bfloat16
    assert add_62.is_cuda and getitem_1619.is_cuda

    device = add_62.device

    # Stage A: Gather weight into a contiguous [4096] buffer.
    # getitem_1619 has shape [8, 512] with strides [27264000, 1]
    rows_w, cols_w = getitem_1619.shape  # 8, 512
    n_weight = rows_w * cols_w  # 4096
    stride_row_w, stride_col_w = getitem_1619.stride()

    weight = torch.empty((n_weight,), dtype=torch.bfloat16, device=device)
    BLOCK_W = 1024
    grid_w = (triton.cdiv(n_weight, BLOCK_W),)
    _gather_weight_kernel[grid_w](
        getitem_1619, weight,
        n_weight, stride_row_w, stride_col_w,
        cols_w, BLOCK_W,
    )

    # Stage B: RMS norm over last dim (4096).
    # add_62 shape [1, 8192, 4096] -> treat as [8192, 4096]
    x = add_62.contiguous().view(-1, add_62.shape[-1])
    n_rows, n_cols = x.shape

    out = torch.empty_like(x)

    # n_cols = 4096, fits in a single block
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Choose num_warps based on BLOCK_SIZE
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    grid = (n_rows,)
    _rms_norm_kernel[grid](
        x, weight, out,
        n_rows, n_cols,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Reshape output to [8192, 4096] and return twice as in reference.
    out_view = out.view(8192, 4096)
    return (out_view, out_view)