import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_complex_mul_kernel(
    in_ptr, clone_ptr, out_ptr,
    S, H, D_HALF,
    BLOCK_S: tl.constexpr,
    H_CONST: tl.constexpr,
    D_HALF_CONST: tl.constexpr,
):
    pid_s = tl.program_id(0)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S

    k = tl.arange(0, D_HALF_CONST)
    h = tl.arange(0, H_CONST)

    # Input layout in memory: [S, H, D_HALF*2] contiguous
    # offset = s*(H*D_HALF*2) + h*(D_HALF*2) + 2*k or +2*k+1
    in_base = s_off[:, None, None] * (H_CONST * D_HALF_CONST * 2) + \
              h[None, :, None] * (D_HALF_CONST * 2) + \
              k[None, None, :] * 2
    
    a_r = tl.load(in_ptr + in_base, mask=s_mask[:, None, None], other=0.0).to(tl.float32)
    a_i = tl.load(in_ptr + in_base + 1, mask=s_mask[:, None, None], other=0.0).to(tl.float32)

    # clone layout: [S, D_HALF*2] (broadcast across H)
    c_base = s_off[:, None] * (D_HALF_CONST * 2) + k[None, :] * 2
    b_r = tl.load(clone_ptr + c_base, mask=s_mask[:, None], other=0.0).to(tl.float32)
    b_i = tl.load(clone_ptr + c_base + 1, mask=s_mask[:, None], other=0.0).to(tl.float32)

    b_r_b = b_r[:, None, :]
    b_i_b = b_i[:, None, :]

    out_r = a_r * b_r_b - a_i * b_i_b
    out_i = a_r * b_i_b + a_i * b_r_b

    tl.store(out_ptr + in_base, out_r.to(out_ptr.dtype.element_ty), mask=s_mask[:, None, None])
    tl.store(out_ptr + in_base + 1, out_i.to(out_ptr.dtype.element_ty), mask=s_mask[:, None, None])


def kernel_function(getitem_642, clone_62):
    assert getitem_642.is_cuda and clone_62.is_cuda
    assert getitem_642.dtype == torch.bfloat16
    assert clone_62.dtype == torch.complex64

    S = 8192
    H = 8
    D_HALF = 64

    clone_flat = clone_62.contiguous()
    clone_as_real = torch.view_as_real(clone_flat)

    output = torch.empty((S, H * D_HALF * 2), dtype=torch.bfloat16, device=getitem_642.device)

    BLOCK_S = 8
    grid = (triton.cdiv(S, BLOCK_S),)
    _rotary_complex_mul_kernel[grid](
        getitem_642, clone_as_real, output,
        S, H, D_HALF,
        BLOCK_S=BLOCK_S,
        H_CONST=H,
        D_HALF_CONST=D_HALF,
        num_warps=8,
        num_stages=3,
    )
    return output