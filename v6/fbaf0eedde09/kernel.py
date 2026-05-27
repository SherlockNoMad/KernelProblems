import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, freq_ptr, out_ptr,
    S, H, D,
    x_stride_h, x_stride_s,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    pair_idx = tl.arange(0, BLOCK_D // 2)

    d_even = 2 * pair_idx
    d_odd = 2 * pair_idx + 1

    x_base = pid_h * x_stride_h + pid_s * x_stride_s
    a = tl.load(x_ptr + x_base + d_even).to(tl.float32)
    b = tl.load(x_ptr + x_base + d_odd).to(tl.float32)

    # freq is complex64 stored as pairs of float32, shape [1, S, 1, D/2]
    freq_base = pid_s * (BLOCK_D // 2) * 2
    c = tl.load(freq_ptr + freq_base + 2 * pair_idx)
    d_im = tl.load(freq_ptr + freq_base + 2 * pair_idx + 1)

    # Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    real = a * c - b * d_im
    imag = a * d_im + b * c

    # Output is [S, H*D] in row-major
    out_base = pid_s * (H * D) + pid_h * D
    tl.store(out_ptr + out_base + d_even, real.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_base + d_odd, imag.to(out_ptr.dtype.element_ty))


@triton.jit
def _copy_v_kernel(
    v_ptr, out_ptr,
    S, H, D,
    v_stride_h, v_stride_s,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_idx = tl.arange(0, BLOCK_D)
    v_base = pid_h * v_stride_h + pid_s * v_stride_s
    val = tl.load(v_ptr + v_base + d_idx)

    out_base = pid_s * (H * D) + pid_h * D
    tl.store(out_ptr + out_base + d_idx, val)


def kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643):
    """
    RoPE-like rotary embedding fused kernel.
    - xq: apply rope with _conj_63 to getitem_641 [1, 32, 8192, 128] -> [8192, 4096] bf16
    - xk: apply rope with _conj_62 to getitem_642 [1, 8, 8192, 128] -> [8192, 1024] bf16
    - xv: transpose + view of getitem_643 [1, 8, 8192, 128] -> [8192, 1024] bf16
    """
    assert getitem_641.is_cuda

    S = 8192
    D = 128

    # ----- xq -----
    Hq = 32
    xq_out = torch.empty((S, Hq * D), dtype=torch.bfloat16, device=getitem_641.device)

    # getitem_641 shape [1, 32, 8192, 128], strides [33554432, 128, 4096, 1]
    # We index as [h, s, d]: stride_h=128, stride_s=4096
    x641_stride_h = getitem_641.stride(1)
    x641_stride_s = getitem_641.stride(2)

    # freq _conj_63: shape [1, 8192, 1, 64] complex64, view as float32 last dim 2 -> [1, 8192, 1, 64, 2]
    # contiguous so each s has 64*2 floats
    freq63_f32 = torch.view_as_real(_conj_63).contiguous()  # [1, 8192, 1, 64, 2]

    grid_q = (S, Hq)
    _rope_kernel[grid_q](
        getitem_641, freq63_f32, xq_out,
        S, Hq, D,
        x641_stride_h, x641_stride_s,
        BLOCK_D=D,
    )

    # ----- xk -----
    Hk = 8
    xk_out = torch.empty((S, Hk * D), dtype=torch.bfloat16, device=getitem_642.device)

    x642_stride_h = getitem_642.stride(1)
    x642_stride_s = getitem_642.stride(2)

    freq62_f32 = torch.view_as_real(_conj_62).contiguous()

    grid_k = (S, Hk)
    _rope_kernel[grid_k](
        getitem_642, freq62_f32, xk_out,
        S, Hk, D,
        x642_stride_h, x642_stride_s,
        BLOCK_D=D,
    )

    # ----- xv -----
    Hv = 8
    xv_out = torch.empty((S, Hv * D), dtype=torch.bfloat16, device=getitem_643.device)

    x643_stride_h = getitem_643.stride(1)
    x643_stride_s = getitem_643.stride(2)

    grid_v = (S, Hv)
    _copy_v_kernel[grid_v](
        getitem_643, xv_out,
        S, Hv, D,
        x643_stride_h, x643_stride_s,
        BLOCK_D=D,
    )

    return (xq_out, xk_out, xv_out)