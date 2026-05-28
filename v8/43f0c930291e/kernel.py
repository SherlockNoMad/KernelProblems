import torch
import triton
import triton.language as tl


@triton.jit
def rms_bwd_kernel(
    mm656_ptr, mm658_ptr, mm660_ptr, add1_ptr, w_ptr, rstd_ptr, add215_ptr,
    out0_ptr, dw_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load gradients - match PyTorch behavior: adds happen in bf16
    dY1 = tl.load(mm656_ptr + m * N + cols, mask=mask, other=0.0)
    dY2 = tl.load(mm658_ptr + m * N + cols, mask=mask, other=0.0)
    dY3 = tl.load(mm660_ptr + m * N + cols, mask=mask, other=0.0)
    # Mimic bf16 intermediate adds
    tmp = (dY1.to(tl.float32) + dY2.to(tl.float32)).to(tl.bfloat16)
    dY_bf16 = (tmp.to(tl.float32) + dY3.to(tl.float32)).to(tl.bfloat16)
    dY = dY_bf16.to(tl.float32)

    X = tl.load(add1_ptr + m * N + cols, mask=mask, other=0.0).to(tl.float32)
    W = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + m).to(tl.float32)
    add215 = tl.load(add215_ptr + m * N + cols, mask=mask, other=0.0).to(tl.float32)

    # Normalized x
    x_hat = X * rstd

    dY_W = dY * W
    # sum over features of dY_W * x_hat
    sum_val = tl.sum(dY_W * x_hat, axis=0)
    # dX = rstd * (dY_W - x_hat * sum_val / N)
    dX = rstd * (dY_W - x_hat * (sum_val / N))

    out0 = add215 + dX
    tl.store(out0_ptr + m * N + cols, out0.to(tl.bfloat16), mask=mask)

    # dW = sum over rows of dY * x_hat (accumulated in fp32)
    dw_contrib = dY * x_hat
    tl.atomic_add(dw_ptr + cols, dw_contrib, mask=mask)


def kernel_function(_fused_rms_norm_2_recomputed, mm_656, mm_658, mm_660, add_1, _unsafe_view_450, add_215):
    rstd = _fused_rms_norm_2_recomputed[1]  # [1, 8192, 1] fp32

    M = 8192
    N = 4096

    out0 = torch.empty((8192, 4096), dtype=torch.bfloat16, device=add_1.device)
    dw = torch.zeros((N,), dtype=torch.float32, device=add_1.device)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)
    rms_bwd_kernel[grid](
        mm_656, mm_658, mm_660, add_1, _unsafe_view_450, rstd, add_215,
        out0, dw,
        M, N,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return (out0, dw)