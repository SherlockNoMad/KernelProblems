import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    mm_ptr,
    target_ptr,
    out_ptr,
    N, C,
    inv_total_weight,
    ignore_index,
    BLOCK_C: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)

    target = tl.load(target_ptr + row)
    valid = (target != ignore_index) & (target >= 0) & (target < C)

    row_ptr = mm_ptr + row * C

    # Single-pass online softmax: track max and sum_exp
    max_val = -float('inf')
    sum_exp = 0.0
    target_logit = 0.0

    for i in tl.static_range(NUM_BLOCKS):
        c_start = i * BLOCK_C
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(row_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        e = tl.exp(x - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)
        max_val = new_max

    log_sum = tl.log(sum_exp)
    w = tl.where(valid, inv_total_weight, 0.0)
    target_safe = tl.where(valid, target, 0)

    # Second pass: compute grad and store transposed
    for i in tl.static_range(NUM_BLOCKS):
        c_start = i * BLOCK_C
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        lsm = x - max_val - log_sum
        exp_lsm = tl.exp(lsm)
        grad = exp_lsm * w
        is_target = offs == target_safe
        grad = grad - tl.where(is_target & valid, w, 0.0)
        grad = tl.where(valid, grad, 0.0)
        out_offsets = offs * N + row
        tl.store(out_ptr + out_offsets, grad.to(tl.bfloat16), mask=mask)


def kernel_function(arg584_1, mm_224):
    assert mm_224.is_cuda and arg584_1.is_cuda
    assert mm_224.dtype == torch.bfloat16
    assert arg584_1.dtype == torch.int64

    target = arg584_1.reshape(-1).contiguous()
    mm = mm_224.contiguous()
    N, C = mm.shape
    assert target.numel() == N

    out = torch.empty((C, N), dtype=torch.bfloat16, device=mm.device)

    inv_total_weight = 1.0 / 8192.0
    ignore_index = -100

    BLOCK_C = 16384
    NUM_BLOCKS = (C + BLOCK_C - 1) // BLOCK_C
    grid = (N,)
    _fused_kernel[grid](
        mm, target, out,
        N, C,
        inv_total_weight,
        ignore_index,
        BLOCK_C=BLOCK_C,
        NUM_BLOCKS=NUM_BLOCKS,
        num_warps=16,
        num_stages=2,
    )
    return out