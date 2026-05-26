import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fused_fwd_bwd_kernel(
    mm223_ptr, add62_ptr, mm226_ptr, weight_ptr,
    grad_input_ptr, temp_gw_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    """
    Fused per-row kernel that computes:
      1) x = add_62_recomputed + mm_223 (bf16 add)
      2) rstd = 1 / sqrt(mean(x^2) + eps)
      3) grad_input = rstd*(g*w) - x*rstd^3 * mean(g*w*x)
         where g = mm_226 (grad_out) and w = wait_tensor_970 (weight)
      4) per-row contribution to grad_weight: g * x * rstd (fp32, stored to temp)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x1 = tl.load(mm223_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(add62_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # fused add (matches the bf16 add in the reference: cast to bf16 then back)
    x_bf16 = (x1 + x2)
    # Mimic bf16 storage round-trip for add_tensor
    x_bf16 = x_bf16.to(tl.bfloat16).to(tl.float32)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(mm226_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # rms statistics
    sum_sq = tl.sum(x_bf16 * x_bf16, axis=0)
    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # c = sum(g * w * x) / N
    gwx = g * w * x_bf16
    c = tl.sum(gwx, axis=0) / N

    # dL/dx = rstd * g * w - x * rstd^3 * c
    grad_input = rstd * g * w - x_bf16 * (rstd * rstd * rstd) * c

    tl.store(grad_input_ptr + row * N + cols,
             grad_input.to(tl.bfloat16), mask=mask)

    # Per-row contribution for grad_weight: g * (x * rstd)
    temp = g * x_bf16 * rstd
    tl.store(temp_gw_ptr + row * N + cols, temp, mask=mask)


@triton.jit
def _reduce_grad_weight_kernel(
    temp_ptr, gw_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Reduce temp [M, N] along M into grad_weight [N] (fp32)."""
    pid = tl.program_id(0)
    col_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = col_offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        ptrs = temp_ptr + m_offs[:, None] * N + col_offsets[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Reference returns bf16 grad_weight which is then cast to fp32.
    # Emulate that round-trip for numerical parity.
    acc_bf16 = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(gw_ptr + col_offsets, acc_bf16, mask=mask_n)


def kernel_function(mm_223, add_62_recomputed, mm_226, wait_tensor_970):
    """
    Fused implementation of:
        x = add_62_recomputed + view(mm_223)
        rstd = 1/sqrt(mean(x^2)+eps)
        grad_out = view(mm_226)
        grad_input, grad_weight = rms_norm_backward(grad_out, x, weight, rstd)
        return (grad_input_bf16, grad_weight_fp32)

    Fusion stages combined in `_rms_fused_fwd_bwd_kernel`:
      - bf16 elementwise add (mm_223 + add_62_recomputed)
      - RMSNorm forward statistics (mean of squares, rstd)
      - RMSNorm backward grad_input
      - Materialize per-row contribution to grad_weight (g*x*rstd)
    A second small kernel reduces those contributions along the batch dim.
    """
    assert mm_223.is_cuda and add_62_recomputed.is_cuda
    assert mm_226.is_cuda and wait_tensor_970.is_cuda
    assert mm_223.dtype == torch.bfloat16
    assert add_62_recomputed.dtype == torch.bfloat16
    assert mm_226.dtype == torch.bfloat16
    assert wait_tensor_970.dtype == torch.bfloat16

    # Shapes
    # mm_223: [8192, 4096] -> view [1,8192,4096]
    # add_62_recomputed: [1,8192,4096]
    # mm_226: [8192,4096] -> view [1,8192,4096]
    # wait_tensor_970: [4096]
    N = wait_tensor_970.numel()  # 4096
    M = mm_223.numel() // N      # 8192
    assert add_62_recomputed.numel() == M * N
    assert mm_226.numel() == M * N

    # Ensure contiguous for flat indexing
    mm_223_c = mm_223.contiguous()
    add_c = add_62_recomputed.contiguous()
    mm_226_c = mm_226.contiguous()
    w_c = wait_tensor_970.contiguous()

    # Allocate outputs
    grad_input = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_223.device)
    temp_gw = torch.empty((M, N), dtype=torch.float32, device=mm_223.device)
    grad_weight_fp32 = torch.empty((N,), dtype=torch.float32, device=mm_223.device)

    eps = 1e-5
    BLOCK_N = triton.next_power_of_2(N)  # 4096

    grid_fwd = (M,)
    _rms_fused_fwd_bwd_kernel[grid_fwd](
        mm_223_c, add_c, mm_226_c, w_c,
        grad_input, temp_gw,
        M, N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    # Reduce temp -> grad_weight
    BLOCK_N_RED = 256
    BLOCK_M_RED = 64
    grid_red = (triton.cdiv(N, BLOCK_N_RED),)
    _reduce_grad_weight_kernel[grid_red](
        temp_gw, grad_weight_fp32,
        M, N,
        BLOCK_M=BLOCK_M_RED, BLOCK_N=BLOCK_N_RED,
        num_warps=4,
    )

    return (grad_input, grad_weight_fp32)