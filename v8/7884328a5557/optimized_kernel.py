import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    mm_ptr,
    add_ptr,
    weight_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * N
    offs = tl.arange(0, BLOCK_SIZE)

    mm_vals = tl.load(mm_ptr + row_off + offs)
    add_vals = tl.load(add_ptr + row_off + offs)
    w = tl.load(weight_ptr + offs)

    summed = mm_vals + add_vals
    summed_f = summed.to(tl.float32)

    sq = summed_f * summed_f
    mean_sq = tl.sum(sq, axis=0) / N
    inv_rms = tl.rsqrt(mean_sq + eps)

    out = summed_f * inv_rms
    out = out.to(tl.bfloat16) * w

    tl.store(out_ptr + row_off + offs, out)


def kernel_function(mm_220, add_61, _unsafe_view_981):
    M = 8192
    N = 4096
    eps = 1e-5

    add_c = add_61.view(M, N)

    out = torch.empty((M, N), dtype=mm_220.dtype, device=mm_220.device)

    grid = (M,)
    _fused_add_rmsnorm_kernel[grid](
        mm_220, add_c, _unsafe_view_981, out,
        N, eps,
        BLOCK_SIZE=N,
        num_warps=8,
        num_stages=3,
    )

    return (out, out)