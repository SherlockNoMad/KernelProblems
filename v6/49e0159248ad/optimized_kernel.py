import torch
import triton
import triton.language as tl


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
    x_ptr,
    target_ptr,
    total_weight_ptr,
    out_ptr,
    N, C,
    inv_N,
    ignore_index,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)

    target = tl.load(target_ptr + row)
    total_weight = tl.load(total_weight_ptr).to(tl.float32)

    safe_tw = tl.where(total_weight > 0, total_weight, 1.0)
    scalar = -inv_N / safe_tw

    is_valid = target != ignore_index

    row_off = row * C

    max_val = -float('inf')
    sum_exp = 0.0
    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(x_ptr + row_off + offs, mask=mask, other=-float('inf'),
                    eviction_policy='evict_last').to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        e = tl.exp(x - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(e, axis=0)
        max_val = new_max

    log_sum_exp = tl.log(sum_exp) + max_val

    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0,
                    eviction_policy='evict_first').to(tl.float32)
        softmax = tl.exp(x - log_sum_exp)
        is_target = offs == target
        grad = scalar * (is_target.to(tl.float32) - softmax)
        grad = tl.where(is_valid, grad, 0.0)
        tl.store(out_ptr + row_off + offs, grad.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(arg584_1, mm_224):
    assert arg584_1.is_cuda and mm_224.is_cuda
    assert mm_224.dtype == torch.bfloat16

    target_flat = arg584_1.reshape(-1).contiguous()
    x = mm_224.contiguous()
    N, C = x.shape

    ignore_index = -100
    inv_N = 1.0 / float(N)

    total_weight = torch.zeros(1, dtype=torch.int32, device=x.device)

    BLOCK_T = 1024
    grid_t = (triton.cdiv(N, BLOCK_T),)
    _count_valid_kernel[grid_t](target_flat, total_weight, N, ignore_index, BLOCK_SIZE=BLOCK_T)

    out = torch.empty_like(x)

    BLOCK_C = 16384
    grid = (N,)
    _fused_logsoftmax_nll_bwd_kernel[grid](
        x, target_flat, total_weight, out,
        N, C,
        inv_N,
        ignore_index,
        BLOCK_C=BLOCK_C,
        num_warps=16,
        num_stages=2,
    )

    return out