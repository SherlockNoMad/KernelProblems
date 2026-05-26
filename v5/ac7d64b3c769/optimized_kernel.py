import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fused_fwd_bwd_kernel(
    mm223_ptr, add62_ptr, mm226_ptr, weight_ptr,
    grad_input_ptr, grad_weight_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x1 = tl.load(mm223_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(add62_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x_bf16 = (x1 + x2)
    x_bf16 = x_bf16.to(tl.bfloat16).to(tl.float32)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(mm226_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x_bf16 * x_bf16, axis=0)
    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    gw = g * w
    c = tl.sum(gw * x_bf16, axis=0) / N

    grad_input = rstd * gw - x_bf16 * (rstd * rstd * rstd) * c

    tl.store(grad_input_ptr + row * N + cols,
             grad_input.to(tl.bfloat16), mask=mask)

    gw_contrib = g * x_bf16 * rstd
    tl.atomic_add(grad_weight_ptr + cols, gw_contrib, mask=mask)


def kernel_function(mm_223, add_62_recomputed, mm_226, wait_tensor_970):
    assert mm_223.is_cuda and add_62_recomputed.is_cuda
    assert mm_226.is_cuda and wait_tensor_970.is_cuda

    N = wait_tensor_970.numel()
    M = mm_223.numel() // N
    assert add_62_recomputed.numel() == M * N
    assert mm_226.numel() == M * N

    mm_223_c = mm_223.contiguous()
    add_c = add_62_recomputed.contiguous()
    mm_226_c = mm_226.contiguous()
    w_c = wait_tensor_970.contiguous()

    grad_input = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_223.device)
    grad_weight_fp32 = torch.zeros((N,), dtype=torch.float32, device=mm_223.device)

    eps = 1e-5
    BLOCK_N = triton.next_power_of_2(N)

    grid_fwd = (M,)
    _rms_fused_fwd_bwd_kernel[grid_fwd](
        mm_223_c, add_c, mm_226_c, w_c,
        grad_input, grad_weight_fp32,
        M, N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return (grad_input, grad_weight_fp32)