import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    mm_ptr,        # [M, N]
    add_ptr,       # [M, N]
    weight_ptr,    # [N]
    out_ptr,       # [M, N]
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel performing:
      1) tmp = add + mm   (elementwise)
      2) rms_norm(tmp, weight, eps) -> out
    Each program handles one row of length N.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    row_off = row * N
    mm_vals = tl.load(mm_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)

    summed = add_vals + mm_vals

    # RMS norm: x / sqrt(mean(x^2) + eps) * weight
    sq = summed * summed
    sq = tl.where(mask, sq, 0.0)
    mean_sq = tl.sum(sq, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = summed * inv_rms * w

    tl.store(out_ptr + row_off + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_220, add_61, _unsafe_view_981):
    """
    Fused implementation of:
       _unsafe_view = mm_220.view(1, 8192, 4096)
       add = add_61 + _unsafe_view
       out = rms_norm(add, [4096], weight=_unsafe_view_981, eps=1e-5)
       return (out.view(8192, 4096), out.view(8192, 4096))

    Fusion: elementwise add + RMS norm reduction + scale, all in one kernel.
    """
    assert mm_220.is_cuda and add_61.is_cuda and _unsafe_view_981.is_cuda
    assert mm_220.shape == (8192, 4096)
    assert add_61.shape == (1, 8192, 4096)
    assert _unsafe_view_981.shape == (4096,)

    M = 8192
    N = 4096
    eps = 1e-5

    mm_c = mm_220.contiguous()
    add_c = add_61.contiguous().view(M, N)
    w_c = _unsafe_view_981.contiguous()

    out = torch.empty((M, N), dtype=mm_220.dtype, device=mm_220.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _fused_add_rmsnorm_kernel[grid](
        mm_c, add_c, w_c, out,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return (out, out)