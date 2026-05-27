import torch
import triton
import triton.language as tl


@triton.jit
def _fused_logsoftmax_nll_kernel(
    logits_ptr,
    targets_ptr,
    partial_loss_ptr,
    partial_weight_ptr,
    N, V,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    target = tl.load(targets_ptr + row)
    is_valid = target != ignore_index

    row_start = row * V
    max_val = -float("inf")
    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(logits_ptr + row_start + offs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        x = tl.load(logits_ptr + row_start + offs, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_val

    safe_target = tl.where(is_valid, target, 0)
    target_logit = tl.load(logits_ptr + row_start + safe_target).to(tl.float32)
    log_prob = target_logit - log_sum_exp
    loss_contrib = -log_prob
    loss_contrib = tl.where(is_valid, loss_contrib, 0.0)
    weight = tl.where(is_valid, 1.0, 0.0)

    tl.store(partial_loss_ptr + row, loss_contrib)
    tl.store(partial_weight_ptr + row, weight)


@triton.jit
def _partial_reduce_kernel(
    partial_loss_ptr,
    partial_weight_ptr,
    block_loss_ptr,
    block_weight_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_N
    offs = start + tl.arange(0, BLOCK_N)
    mask = offs < N
    l = tl.load(partial_loss_ptr + offs, mask=mask, other=0.0)
    w = tl.load(partial_weight_ptr + offs, mask=mask, other=0.0)
    loss_sum = tl.sum(l, axis=0)
    weight_sum = tl.sum(w, axis=0)
    tl.store(block_loss_ptr + pid, loss_sum)
    tl.store(block_weight_ptr + pid, weight_sum)


@triton.jit
def _final_reduce_kernel(
    block_loss_ptr,
    block_weight_ptr,
    loss_out_ptr,
    weight_out_ptr,
    NUM_BLOCKS,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_BLOCKS
    l = tl.load(block_loss_ptr + offs, mask=mask, other=0.0)
    w = tl.load(block_weight_ptr + offs, mask=mask, other=0.0)
    loss_total = tl.sum(l, axis=0)
    weight_total = tl.sum(w, axis=0)
    tl.store(loss_out_ptr, loss_total / divisor)
    tl.store(weight_out_ptr, weight_total)


def kernel_function(mm_224, view_805):
    assert mm_224.is_cuda and view_805.is_cuda
    if mm_224.dim() == 2:
        N, V = mm_224.shape
    else:
        V = mm_224.shape[-1]
        N = mm_224.numel() // V
    logits = mm_224.contiguous().view(N, V)
    targets = view_805.contiguous()
    assert targets.numel() == N

    partial_loss = torch.empty(N, device=logits.device, dtype=torch.float32)
    partial_weight = torch.empty(N, device=logits.device, dtype=torch.float32)

    BLOCK_V = 4096
    grid = (N,)
    _fused_logsoftmax_nll_kernel[grid](
        logits, targets, partial_loss, partial_weight,
        N, V,
        -100,
        BLOCK_V=BLOCK_V,
        num_warps=16,
    )

    BLOCK_N = 256
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N
    block_loss = torch.empty(num_blocks, device=logits.device, dtype=torch.float32)
    block_weight = torch.empty(num_blocks, device=logits.device, dtype=torch.float32)

    _partial_reduce_kernel[(num_blocks,)](
        partial_loss, partial_weight, block_loss, block_weight,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    loss_out = torch.empty((), device=logits.device, dtype=torch.float32)
    weight_out = torch.empty((), device=logits.device, dtype=torch.float32)

    # next pow2 >= num_blocks
    BS = 1
    while BS < num_blocks:
        BS *= 2
    BS = max(BS, 32)

    _final_reduce_kernel[(1,)](
        block_loss, block_weight, loss_out, weight_out,
        num_blocks, 8192.0,
        BLOCK_SIZE=BS,
        num_warps=4,
    )

    return (loss_out, weight_out)