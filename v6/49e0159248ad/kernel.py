import torch
import triton
import triton.language as tl


# Fused kernel computes:
# 1. log_softmax over each row of the input (mm_224 viewed as [8192, 128256], cast to fp32)
# 2. NLL loss forward to compute total_weight (number of non-ignored targets)
# 3. log_softmax_backward = softmax(x) * sum(grad) - grad  (but here grad is sparse,
#    so the gradient input to log_softmax_backward is: g[i, target[i]] = -1/8192/total_weight
#    if target != ignore_index else 0)
# 4. The output is cast back to bf16.
#
# For row i with target t (ignore_index = -100):
#   if t == -100: grad_input[i, :] = 0
#   else:
#     scalar = -1/8192 / total_weight
#     g[i, j] = scalar if j == t else 0
#     sum_g = scalar
#     log_softmax_back[i, j] = g[i, j] - exp(log_softmax[i, j]) * sum_g
#                            = g[i, j] - softmax[i, j] * scalar
#
# So for each row, we need:
#   - max of row (for stable softmax)
#   - sum of exp(x - max) -> log_sum_exp
#   - then for each element: softmax = exp(x - max - lse)
#   - output = (1 if j==t else 0)*scalar - softmax*scalar
#         = scalar * ((j==t ? 1 : 0) - softmax)
#
# We need total_weight = count of targets != ignore_index. We compute this first in
# a small kernel, then run the main per-row fused kernel.


@triton.jit
def _count_valid_kernel(target_ptr, out_ptr, N, ignore_index, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    t = tl.load(target_ptr + offs, mask=mask, other=ignore_index)
    valid = (t != ignore_index) & mask
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.atomic_add(out_ptr, count)


@triton.jit
def _fused_logsoftmax_nll_bwd_kernel(
    x_ptr,           # [N, C] bf16 input (mm_224 reshaped)
    target_ptr,      # [N] int64
    total_weight_ptr,# [1] int32 (count of valid targets)
    out_ptr,         # [N, C] bf16 output
    N, C,
    inv_N,           # 1/8192.0 as a float
    ignore_index,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    target = tl.load(target_ptr + row)
    total_weight = tl.load(total_weight_ptr).to(tl.float32)

    # Compute scalar = -inv_N / total_weight  (only if target valid)
    # If total_weight == 0, avoid division by zero; in that case output is 0 anyway.
    safe_tw = tl.where(total_weight > 0, total_weight, 1.0)
    scalar = -inv_N / safe_tw

    is_valid = target != ignore_index

    row_off = row * C

    # Pass 1: compute max
    max_val = -float('inf')
    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(x_ptr + row_off + offs, mask=mask, other=-float('inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: compute sum exp
    sum_exp = 0.0
    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(x_ptr + row_off + offs, mask=mask, other=-float('inf')).to(tl.float32)
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_val

    # Pass 3: compute output
    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        softmax = tl.exp(x - log_sum_exp)
        # grad_input[j] = scalar * ((j==target ? 1 : 0) - softmax)  if valid else 0
        is_target = offs == target
        grad = scalar * (is_target.to(tl.float32) - softmax)
        grad = tl.where(is_valid, grad, 0.0)
        tl.store(out_ptr + row_off + offs, grad.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(arg584_1, mm_224):
    """
    Fused kernel that computes the full forward+backward of:
      log_softmax -> nll_loss -> div(loss, N) -> backprop with grad_output = 1
    
    Fused stages:
      - Count valid targets (for total_weight)
      - Per-row: max, log-sum-exp, softmax, NLL gradient, log_softmax backward,
        and bf16 cast — all in a single kernel pass over each row.
    
    Inputs:
      arg584_1: [1, 8192] int64 target indices
      mm_224:   [8192, 128256] bf16 logits
    Returns:
      [8192, 128256] bf16 gradient tensor
    """
    assert arg584_1.is_cuda and mm_224.is_cuda
    assert mm_224.dtype == torch.bfloat16

    # Flatten target to [N]
    target_flat = arg584_1.reshape(-1).contiguous()
    x = mm_224.contiguous()
    N, C = x.shape
    assert target_flat.numel() == N

    ignore_index = -100
    inv_N = 1.0 / float(N)

    # Allocate total_weight counter
    total_weight = torch.zeros(1, dtype=torch.int32, device=x.device)

    BLOCK_T = 1024
    grid_t = (triton.cdiv(N, BLOCK_T),)
    _count_valid_kernel[grid_t](target_flat, total_weight, N, ignore_index, BLOCK_SIZE=BLOCK_T)

    # Output
    out = torch.empty_like(x)

    # Choose BLOCK_C. C = 128256. Use 4096 as block size for the column loop.
    BLOCK_C = 4096
    grid = (N,)
    _fused_logsoftmax_nll_bwd_kernel[grid](
        x, target_flat, total_weight, out,
        N, C,
        inv_N,
        ignore_index,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )

    return out