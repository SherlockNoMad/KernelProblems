import torch
import triton
import triton.language as tl

# Fused kernel:
# - out0 = view(mm_662) * silu_recomputed   (reshaped to [8192, 14336])
# - tmp  = view(mm_662) * _unsafe_view_5_recomputed
# - out1 = silu_backward(tmp, _unsafe_view_4_recomputed)
#   where silu_backward(grad, x) = grad * sigmoid(x) * (1 + x * (1 - sigmoid(x)))

@triton.jit
def _fused_kernel(
    mm_ptr, silu_ptr, v5_ptr, v4_ptr,
    out0_ptr, out1_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = tl.load(silu_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    v5 = tl.load(v5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    v4 = tl.load(v4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Output 0: mm * silu
    out0 = mm * silu

    # Output 1: silu_backward(mm * v5, v4)
    grad = mm * v5
    sig = tl.sigmoid(v4)
    # d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    dsilu = sig * (1.0 + v4 * (1.0 - sig))
    out1 = grad * dsilu

    tl.store(out0_ptr + offsets, out0.to(out0_ptr.dtype.element_ty), mask=mask)
    tl.store(out1_ptr + offsets, out1.to(out1_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_662, silu_recomputed, _unsafe_view_5_recomputed, _unsafe_view_4_recomputed):
    """
    Fused implementation of:
      view_default = view(mm_662, [1, 8192, 14336])
      mul_tensor = view_default * silu_recomputed
      view_default_1 = view(mul_tensor, [8192, 14336])
      mul_tensor_1 = view_default * _unsafe_view_5_recomputed
      silu_backward_default = silu_backward(mul_tensor_1, _unsafe_view_4_recomputed)
      view_default_2 = view(silu_backward_default, [8192, 14336])
    Returns (view_default_1, view_default_2).

    All operations are elementwise on tensors with 8192*14336 elements, so we fuse
    them into a single Triton kernel pass over the data.
    """
    assert mm_662.is_cuda and silu_recomputed.is_cuda
    assert _unsafe_view_5_recomputed.is_cuda and _unsafe_view_4_recomputed.is_cuda

    # Ensure contiguous for flat indexing
    mm_flat = mm_662.contiguous().view(-1)
    silu_flat = silu_recomputed.contiguous().view(-1)
    v5_flat = _unsafe_view_5_recomputed.contiguous().view(-1)
    v4_flat = _unsafe_view_4_recomputed.contiguous().view(-1)

    n_elements = mm_flat.numel()
    assert silu_flat.numel() == n_elements
    assert v5_flat.numel() == n_elements
    assert v4_flat.numel() == n_elements

    out0 = torch.empty((8192, 14336), dtype=mm_662.dtype, device=mm_662.device)
    out1 = torch.empty((8192, 14336), dtype=mm_662.dtype, device=mm_662.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_kernel[grid](
        mm_flat, silu_flat, v5_flat, v4_flat,
        out0.view(-1), out1.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return (out0, out1)