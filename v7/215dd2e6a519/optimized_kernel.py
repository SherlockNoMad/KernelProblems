import torch
import triton
import triton.language as tl


# Fused kernel: 
# 1) Gather weight from strided bf16 tensor `getitem_1614` (shape [8,512], stride [27264000,1])
#    reshaped contiguously to [4096] (i.e., flatten in row-major order).
# 2) Compute RMSNorm along last dim (4096) of `add_61` (shape [1,8192,4096]).
# 3) Output is viewed as [8192, 4096] and returned three times (same storage).
#
# All math (gather + rms reduction + normalize + scale) happens in one Triton kernel.

@triton.jit
def _rmsnorm_fused_kernel(
    x_ptr,           # add_61, contiguous, shape [M, N]
    w_src_ptr,       # getitem_1614 base pointer (bf16), strided
    out_ptr,         # output, shape [M, N], contiguous
    M, N,
    w_stride0, w_stride1,   # strides for w_src in elements
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load x for this row
    x_ptrs = x_ptr + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean of squares
    sumsq = tl.sum(x * x, axis=0)
    mean_sq = sumsq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Gather weight: linear index i in [0,N) -> (i // 512, i % 512) into strided tensor
    # since logical shape is [8,512] reshaped to [4096]
    w_row = offs // 512
    w_col = offs % 512
    w_ptrs = w_src_ptr + w_row * w_stride0 + w_col * w_stride1
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    y = x * rstd * w

    out_ptrs = out_ptr + row * N + offs
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(getitem_1614, add_61):
    """
    Fused RMSNorm with gathered strided weight.
    
    Fused stages:
      - Gather weight from strided bf16 tensor into logical [4096]
      - Compute RMSNorm over last dim of add_61 ([1, 8192, 4096])
      - Apply weight scaling
      - Produce output viewed as [8192, 4096] (returned 3x)
    """
    assert add_61.is_cuda and getitem_1614.is_cuda
    assert add_61.dtype == torch.bfloat16
    assert getitem_1614.dtype == torch.bfloat16

    # add_61 shape [1, 8192, 4096]; we treat it as [M, N] = [8192, 4096]
    # ensure contiguous
    x = add_61.contiguous().view(-1, add_61.shape[-1])
    M, N = x.shape
    assert N == 4096

    # weight source: shape [8, 512], stride [27264000, 1] (in elements)
    assert getitem_1614.shape == (8, 512)
    w_stride0, w_stride1 = getitem_1614.stride()

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    BLOCK_N = 4096  # N is exactly 4096, power of 2
    grid = (M,)

    _rmsnorm_fused_kernel[grid](
        x, getitem_1614, out,
        M, N,
        w_stride0, w_stride1,
        1e-5,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )

    # Three views of the same output
    v1 = out.view(8192, 4096)
    v2 = out.view(8192, 4096)
    v3 = out.view(8192, 4096)
    return v1, v2, v3