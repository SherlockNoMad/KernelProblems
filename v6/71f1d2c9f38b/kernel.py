import torch
import triton
import triton.language as tl


@triton.jit
def _nll_logsoftmax_kernel(
    logits_ptr,      # [N, C] bfloat16
    target_ptr,      # [N] int64
    loss_sum_ptr,    # [1] float32 - accumulated loss sum
    total_weight_ptr,# [1] float32 - total weight (= N for valid targets)
    N, C,
    ignore_index,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel: for each row n in [0, N):
      1. Load target t = target_ptr[n]
      2. If t == ignore_index, skip
      3. Compute log_softmax(logits[n])[t] = logits[n,t] - max - log(sum exp(logits[n] - max))
      4. Atomically add -log_softmax_value to loss_sum, and 1.0 to total_weight
    """
    n = tl.program_id(0)
    if n >= N:
        return

    t = tl.load(target_ptr + n)
    # If ignore index, skip entirely
    if t == ignore_index:
        return

    # Compute max over C with tiles
    offs = tl.arange(0, BLOCK_C)
    row_ptr = logits_ptr + n * C

    # Pass 1: find max
    max_val = -float("inf")
    for c_start in range(0, C, BLOCK_C):
        idx = c_start + offs
        mask = idx < C
        x = tl.load(row_ptr + idx, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: sum of exp(x - max)
    sum_exp = 0.0
    for c_start in range(0, C, BLOCK_C):
        idx = c_start + offs
        mask = idx < C
        x = tl.load(row_ptr + idx, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_val

    # Load logit at target
    target_logit = tl.load(row_ptr + t).to(tl.float32)
    log_prob = target_logit - log_sum_exp
    loss = -log_prob

    # Atomic accumulate
    tl.atomic_add(loss_sum_ptr, loss)
    tl.atomic_add(total_weight_ptr, 1.0)


@triton.jit
def _div_kernel(loss_sum_ptr, total_weight_out_ptr, div_out_ptr, divisor):
    """Compute div = loss_sum / divisor and copy total_weight."""
    ls = tl.load(loss_sum_ptr)
    tl.store(div_out_ptr, ls / divisor)


def kernel_function(arg584_1, mm_224):
    """
    Fused log_softmax + nll_loss + divide.

    Stages fused inside the Triton kernel:
      - Reshape (handled via stride/shape interpretation, no copy).
      - Cast bf16 -> fp32 (inside kernel via .to(tl.float32)).
      - Per-row log_softmax (two passes over C: max, then sum-exp).
      - Pick target index, compute -log_prob.
      - Atomic accumulation into a global loss sum and total weight count.
      - Final scalar division loss / 8192.0.
    """
    assert arg584_1.is_cuda and mm_224.is_cuda
    # arg584_1: [1, 8192] int64 -> view as [8192]
    targets = arg584_1.view(-1)
    # mm_224: [8192, 128256] bf16 -> view [1,8192,128256] -> [8192,128256], same memory
    logits = mm_224.view(-1, mm_224.shape[-1])

    N, C = logits.shape
    assert targets.numel() == N

    device = logits.device
    loss_sum = torch.zeros(1, dtype=torch.float32, device=device)
    total_weight = torch.zeros(1, dtype=torch.float32, device=device)

    BLOCK_C = 2048
    ignore_index = -100

    grid = (N,)
    _nll_logsoftmax_kernel[grid](
        logits, targets,
        loss_sum, total_weight,
        N, C,
        ignore_index,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )

    div_out = torch.empty(1, dtype=torch.float32, device=device)
    _div_kernel[(1,)](loss_sum, total_weight, div_out, 8192.0)

    # Return scalar tensors as the reference does (0-d)
    return div_out.view(()), total_weight.view(())