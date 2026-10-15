"""Fused feed-forward gradients, preserving every reference BF16 rounding."""

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _silu_backward_kernel(mm_ptr, silu_ptr, v5_ptr, v4_ptr,
                          out0_ptr, out1_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    mm = tl.load(mm_ptr + offsets, mask, other=0).to(tl.float32)
    silu = tl.load(silu_ptr + offsets, mask, other=0).to(tl.float32)
    v5 = tl.load(v5_ptr + offsets, mask, other=0).to(tl.float32)
    v4 = tl.load(v4_ptr + offsets, mask, other=0).to(tl.float32)
    out0 = mm * silu
    grad = (mm * v5).to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-v4))
    # CUDA's reference contracts the derivative's multiply and add, while
    # retaining the left-to-right order of its two remaining multiplies.
    derivative = tl.fma(v4, 1.0 - sigmoid, 1.0)
    out1 = (grad * sigmoid) * derivative
    tl.store(out0_ptr + offsets, out0, mask)
    tl.store(out1_ptr + offsets, out1, mask)


def kernel_function(mm_662, silu_recomputed, _unsafe_view_5_recomputed,
                    _unsafe_view_4_recomputed):
    out0 = torch.empty_like(mm_662)
    out1 = torch.empty_like(mm_662)
    n_elements = mm_662.numel()
    _silu_backward_kernel[(triton.cdiv(n_elements, 1024),)](
        mm_662, silu_recomputed, _unsafe_view_5_recomputed,
        _unsafe_view_4_recomputed, out0, out1, n_elements,
        BLOCK_SIZE=1024, num_warps=4, enable_fp_fusion=False,
    )
    return out0, out1
