"""SiLU and multiplication with the reference's intervening BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask, other=0).to(tl.float32)
    # SFU exp and ATen's expf produce identical rounded BF16 SiLU values for
    # all 65,536 BF16 inputs, including subnormals, infinities, and NaNs.
    # Keep the correctly rounded division and the materialized BF16 boundary.
    silu = tl.div_rn(a, 1.0 + tl.exp(-a))
    silu = silu.to(tl.bfloat16).to(tl.float32)
    result = silu * b
    tl.store(out_ptr + offsets, result, mask)


def kernel_function(mm_4, mm_5):
    out = torch.empty_like(mm_4)
    n_elements = mm_4.numel()
    _silu_mul_kernel[(triton.cdiv(n_elements, 1024),)](
        mm_4, mm_5, out, n_elements, BLOCK_SIZE=1024, num_warps=4,
        enable_fp_fusion=False,
    )
    return out
