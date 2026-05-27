import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_fused_kernel(
    add_ptr, mm_ptr, w_ptr, out_ptr,
    w_stride0,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_off = row * N
    a = tl.load(add_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mm_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    x = a + m

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    # gather weight from strided wait_tensor: 4096 elements = 8 rows * 512
    w_row = cols // 512
    w_col = cols % 512
    w = tl.load(w_ptr + w_row * w_stride0 + w_col, mask=mask, other=0.0).to(tl.float32)

    y = x * rstd * w

    tl.store(out_ptr + row_off + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr, dst_ptr,
    src_stride0,
    D1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D1
    x = tl.load(src_ptr + row * src_stride0 + cols, mask=mask)
    tl.store(dst_ptr + row * D1 + cols, x, mask=mask)


def kernel_function(mm_223, add_62, wait_tensor_969):
    total_per_row = 65667584
    D1 = 65667072

    # View wait_tensor as (8, total_per_row) for the rms weight strided access
    wt_view = torch.as_strided(wait_tensor_969, size=(8, 512), stride=(total_per_row, 1), storage_offset=0)

    # Second slice copy to contiguous (128256, 4096)
    out2 = torch.empty(128256, 4096, dtype=torch.bfloat16, device=wait_tensor_969.device)
    out2_flat = out2.view(8, D1)
    BLOCK2 = 4096
    second_view = torch.as_strided(wait_tensor_969, size=(8, D1), stride=(total_per_row, 1), storage_offset=512)
    grid2 = (triton.cdiv(D1, BLOCK2), 8)
    _strided_copy_kernel[grid2](
        second_view, out2_flat,
        second_view.stride(0),
        D1,
        BLOCK=BLOCK2,
        num_warps=8,
    )

    # Fused add + RMSNorm, reading rms weight directly from wait_tensor (no copy kernel)
    M, N = 8192, 4096
    out1 = torch.empty(M, N, dtype=torch.bfloat16, device=mm_223.device)

    add_flat = add_62.view(M, N)
    mm_flat = mm_223.view(M, N)

    _add_rmsnorm_fused_kernel[(M,)](
        add_flat, mm_flat, wt_view, out1,
        wt_view.stride(0),
        N, 1e-5,
        BLOCK_SIZE=4096,
        num_warps=8,
    )

    return out1, out2