import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, freq_ptr, out_ptr,
    S, H, D,
    x_stride_h, x_stride_s,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_HALF: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    pair_idx = tl.arange(0, D_HALF)  # [D_HALF]

    s_mask = s_offs < S
    h_mask = h_offs < H

    # x indexed as [h, s, d]: base = h*stride_h + s*stride_s + d
    # Shape: [BLOCK_S, BLOCK_H, D_HALF]
    x_base = (h_offs[None, :, None] * x_stride_h
              + s_offs[:, None, None] * x_stride_s)
    d_even = 2 * pair_idx[None, None, :]
    d_odd = d_even + 1

    mask = s_mask[:, None, None] & h_mask[None, :, None]

    a = tl.load(x_ptr + x_base + d_even, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + x_base + d_odd, mask=mask, other=0.0).to(tl.float32)

    # freq: [1, S, 1, D/2] complex64 as float32 pairs: [S, D_HALF, 2]
    freq_base = s_offs[:, None] * (D_HALF * 2) + 2 * pair_idx[None, :]  # [BLOCK_S, D_HALF]
    c = tl.load(freq_ptr + freq_base, mask=s_mask[:, None], other=0.0)
    d_im = tl.load(freq_ptr + freq_base + 1, mask=s_mask[:, None], other=0.0)

    c_b = c[:, None, :]
    d_b = d_im[:, None, :]

    real = a * c_b - b * d_b
    imag = a * d_b + b * c_b

    # Output [S, H*D] row-major: base = s * (H*D) + h * D + d
    out_base = (s_offs[:, None, None] * (H * D)
                + h_offs[None, :, None] * D)
    tl.store(out_ptr + out_base + d_even, real.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + out_base + d_odd, imag.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_v_kernel(
    v_ptr, out_ptr,
    S, H, D,
    v_stride_h, v_stride_s,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)

    s_mask = s_offs < S
    h_mask = h_offs < H
    mask = s_mask[:, None, None] & h_mask[None, :, None]

    v_base = (h_offs[None, :, None] * v_stride_h
              + s_offs[:, None, None] * v_stride_s
              + d_offs[None, None, :])
    val = tl.load(v_ptr + v_base, mask=mask, other=0.0)

    out_base = (s_offs[:, None, None] * (H * D)
                + h_offs[None, :, None] * D
                + d_offs[None, None, :])
    tl.store(out_ptr + out_base, val, mask=mask)


def kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643):
    S = 8192
    D = 128
    D_HALF = 64

    # ----- xq -----
    Hq = 32
    xq_out = torch.empty((S, Hq * D), dtype=torch.bfloat16, device=getitem_641.device)
    x641_stride_h = getitem_641.stride(1)
    x641_stride_s = getitem_641.stride(2)
    freq63_f32 = torch.view_as_real(_conj_63).contiguous()

    BLOCK_S_Q = 16
    BLOCK_H_Q = 8
    grid_q = (triton.cdiv(S, BLOCK_S_Q), triton.cdiv(Hq, BLOCK_H_Q))
    _rope_kernel[grid_q](
        getitem_641, freq63_f32, xq_out,
        S, Hq, D,
        x641_stride_h, x641_stride_s,
        BLOCK_S=BLOCK_S_Q, BLOCK_H=BLOCK_H_Q, D_HALF=D_HALF,
        num_warps=8,
    )

    # ----- xk -----
    Hk = 8
    xk_out = torch.empty((S, Hk * D), dtype=torch.bfloat16, device=getitem_642.device)
    x642_stride_h = getitem_642.stride(1)
    x642_stride_s = getitem_642.stride(2)
    freq62_f32 = torch.view_as_real(_conj_62).contiguous()

    BLOCK_S_K = 16
    BLOCK_H_K = 8
    grid_k = (triton.cdiv(S, BLOCK_S_K), triton.cdiv(Hk, BLOCK_H_K))
    _rope_kernel[grid_k](
        getitem_642, freq62_f32, xk_out,
        S, Hk, D,
        x642_stride_h, x642_stride_s,
        BLOCK_S=BLOCK_S_K, BLOCK_H=BLOCK_H_K, D_HALF=D_HALF,
        num_warps=8,
    )

    # ----- xv -----
    Hv = 8
    xv_out = torch.empty((S, Hv * D), dtype=torch.bfloat16, device=getitem_643.device)
    x643_stride_h = getitem_643.stride(1)
    x643_stride_s = getitem_643.stride(2)

    BLOCK_S_V = 16
    BLOCK_H_V = 8
    grid_v = (triton.cdiv(S, BLOCK_S_V), triton.cdiv(Hv, BLOCK_H_V))
    _copy_v_kernel[grid_v](
        getitem_643, xv_out,
        S, Hv, D,
        x643_stride_h, x643_stride_s,
        BLOCK_S=BLOCK_S_V, BLOCK_H=BLOCK_H_V, BLOCK_D=D,
        num_warps=8,
    )

    return (xq_out, xk_out, xv_out)