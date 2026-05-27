import torch
import triton
import triton.language as tl

# Fused kernel: per-row computes log_softmax over V classes (in bf16 -> fp32),
# then gathers the target index, and accumulates -log_softmax[target] into a global
# loss accumulator. A second small kernel divides by 8192.0 and computes total weight.
#
# Fused stages:
#   1) _to_copy (bf16 -> fp32)
#   2) _log_softmax along dim=1 (max subtract, exp sum, log)
#   3) nll_loss_forward (gather target, negate, ignore_index handling, reduction=sum)
#   4) div by 8192.0 (final scaling done in epilogue kernel)

@triton.jit
def _fused_logsoftmax_nll_kernel(
    logits_ptr,        # [N, V] bf16
    targets_ptr,       # [N] int64
    partial_loss_ptr,  # [N] fp32 - per-row contribution to loss
    partial_weight_ptr,# [N] fp32 - per-row weight (1 if not ignored else 0)
    N, V,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    target = tl.load(targets_ptr + row)
    is_valid = target != ignore_index

    # Pass 1: compute max
    row_start = row * V
    max_val = -float("inf")
    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(logits_ptr + row_start + offs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: compute sum of exp(x - max)
    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(logits_ptr + row_start + offs, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_val

    # Gather target logit
    # clamp target to [0, V-1] for safe load when ignored
    safe_target = tl.where(is_valid, target, 0)
    target_logit = tl.load(logits_ptr + row_start + safe_target).to(tl.float32)
    log_prob = target_logit - log_sum_exp
    loss_contrib = -log_prob
    loss_contrib = tl.where(is_valid, loss_contrib, 0.0)
    weight = tl.where(is_valid, 1.0, 0.0)

    tl.store(partial_loss_ptr + row, loss_contrib)
    tl.store(partial_weight_ptr + row, weight)


@triton.jit
def _reduce_and_div_kernel(
    partial_loss_ptr,
    partial_weight_ptr,
    loss_out_ptr,         # scalar fp32
    weight_out_ptr,       # scalar fp32
    N,
    divisor,
    BLOCK_N: tl.constexpr,
):
    # Single program reduces all N elements
    pid = tl.program_id(0)
    if pid != 0:
        return

    loss_acc = 0.0
    weight_acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        l = tl.load(partial_loss_ptr + offs, mask=mask, other=0.0)
        w = tl.load(partial_weight_ptr + offs, mask=mask, other=0.0)
        loss_acc += tl.sum(l, axis=0)
        weight_acc += tl.sum(w, axis=0)

    tl.store(loss_out_ptr, loss_acc / divisor)
    tl.store(weight_out_ptr, weight_acc)


def kernel_function(mm_224, view_805):
    """
    Fused implementation of:
      reshape -> _to_copy(fp32) -> log_softmax(dim=1) -> nll_loss(sum) -> div(8192)

    Inputs:
      mm_224: [8192, 128256] bf16 (or any [N*M, V] reshapeable to [1, N, V])
      view_805: [8192] int64 targets
    Returns:
      (div_tensor: fp32 scalar, total_weight: fp32 scalar)
    """
    assert mm_224.is_cuda and view_805.is_cuda
    # The model reshapes to [1, 8192, 128256] then views to [8192, 128256].
    # Treat input as [N, V].
    if mm_224.dim() == 2:
        N, V = mm_224.shape
    else:
        # flatten leading dims
        V = mm_224.shape[-1]
        N = mm_224.numel() // V
    logits = mm_224.contiguous().view(N, V)
    targets = view_805.contiguous()
    assert targets.numel() == N

    partial_loss = torch.empty(N, device=logits.device, dtype=torch.float32)
    partial_weight = torch.empty(N, device=logits.device, dtype=torch.float32)

    BLOCK_V = 1024
    grid = (N,)
    _fused_logsoftmax_nll_kernel[grid](
        logits, targets, partial_loss, partial_weight,
        N, V,
        -100,
        BLOCK_V=BLOCK_V,
        num_warps=8,
    )

    loss_out = torch.empty((), device=logits.device, dtype=torch.float32)
    weight_out = torch.empty((), device=logits.device, dtype=torch.float32)

    BLOCK_N = 1024
    _reduce_and_div_kernel[(1,)](
        partial_loss, partial_weight, loss_out, weight_out,
        N, 8192.0,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return (loss_out, weight_out)