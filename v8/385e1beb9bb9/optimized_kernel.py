import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    mm_ptr, silu_ptr, v5_ptr, v4_ptr,
    out0_ptr, out1_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
    silu = tl.load(silu_ptr + offsets, mask=mask, other=0.0)
    v5 = tl.load(v5_ptr + offsets, mask=mask, other=0.0)
    v4 = tl.load(v4_ptr + offsets, mask=mask, other=0.0)

    mm_f = mm.to(tl.float32)
    v4_f = v4.to(tl.float32)

    out0 = mm_f * silu.to(tl.float32)

    grad = mm_f * v5.to(tl.float32)
    sig = tl.sigmoid(v4_f)
    dsilu = sig * (1.0 + v4_f * (1.0 - sig))
    out1 = grad * dsilu

    tl.store(out0_ptr + offsets, out0.to(out0_ptr.dtype.element_ty), mask=mask)
    tl.store(out1_ptr + offsets, out1.to(out1_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_662, silu_recomputed, _unsafe_view_5_recomputed, _unsafe_view_4_recomputed):
    mm_flat = mm_662.view(-1)
    silu_flat = _unsafe_view_5_recomputed.view(-1) if False else silu_recomputed.view(-1)
    v5_flat = _unsafe_view_5_recomputed.view(-1)
    v4_flat = _unsafe_view_4_recomputed.view(-1)

    n_elements = mm_flat.numel()

    out0 = torch.empty((8192, 14336), dtype=mm_662.dtype, device=mm_662.device)
    out1 = torch.empty((8192, 14336), dtype=mm_662.dtype, device=mm_662.device)

    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_kernel[grid](
        mm_flat, silu_flat, v5_flat, v4_flat,
        out0.view(-1), out1.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        num_stages=2,
    )

    return (out0, out1)