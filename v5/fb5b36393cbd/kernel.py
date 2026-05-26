import torch
import triton
import triton.language as tl


# Fused pipeline:
# Forward: log_softmax(mm_224.float()) over dim=1, then NLL backward with mean reduction
# weight = 1/8192 when target != -100 and target valid, else 0
# total_weight = count of valid targets
# grad_input[i, j] = (-weight[i] * (j == target[i]) - exp(log_softmax[i,j]) * (-weight[i])) ... 
# Actually for log_softmax_backward: grad_in = grad_out - exp(log_softmax) * sum(grad_out, dim=1)
# where grad_out from nll_backward: grad_out[i, target[i]] = -weight[i] if target valid else 0
# sum(grad_out, dim=1) = -weight[i] (if valid) else 0
# So grad_input[i, j] = (-w_i if j==t_i else 0) - exp(lsm[i,j]) * (-w_i)
#                     = exp(lsm[i,j]) * w_i - (w_i if j==t_i else 0)   for valid target
#                     = 0 for invalid target
# Then cast to bf16 and transpose -> output shape [128256, 8192]
# We store directly to transposed layout: out[j, i] = value


@triton.jit
def _fused_kernel(
    mm_ptr,        # bf16 [N, C]
    target_ptr,    # int64 [N]
    out_ptr,       # bf16 [C, N] (transposed)
    N, C,
    inv_total_weight,  # 1/8192.0
    ignore_index,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)  # 0..N-1

    target = tl.load(target_ptr + row)
    valid = (target != ignore_index) & (target >= 0) & (target < C)

    # Pass 1: compute max
    max_val = -float('inf')
    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(mm_ptr + row * C + offs, mask=mask, other=-float('inf')).to(tl.float32)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: compute sum of exp
    sum_exp = 0.0
    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(mm_ptr + row * C + offs, mask=mask, other=-float('inf')).to(tl.float32)
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    log_sum = tl.log(sum_exp)

    # weight for this row
    w = tl.where(valid, inv_total_weight, 0.0)

    # Pass 3: compute grad and store transposed
    # grad[i,j] = exp(lsm[i,j]) * w - (w if j==target else 0) when valid, else 0
    # exp(lsm[i,j]) = exp(x[i,j] - max - log_sum)
    target_safe = tl.where(valid, target, 0)

    for c_start in range(0, C, BLOCK_C):
        offs = c_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        x = tl.load(mm_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        lsm = x - max_val - log_sum
        exp_lsm = tl.exp(lsm)
        grad = exp_lsm * w
        # subtract w at target position
        is_target = offs == target_safe
        grad = grad - tl.where(is_target & valid, w, 0.0)
        # If invalid row, grad = 0
        grad = tl.where(valid, grad, 0.0)
        # Store transposed: out[j, i] = grad[i, j]  -> out_ptr + j * N + i
        out_offsets = offs * N + row
        tl.store(out_ptr + out_offsets, grad.to(tl.bfloat16), mask=mask)


def kernel_function(arg584_1, mm_224):
    """
    Fused log_softmax + NLL loss backward + cast + transpose.
    Returns gradient w.r.t. mm_224 input, transposed to [C, N], in bf16.
    """
    assert mm_224.is_cuda and arg584_1.is_cuda
    assert mm_224.dtype == torch.bfloat16
    assert arg584_1.dtype == torch.int64

    # Flatten target
    target = arg584_1.reshape(-1).contiguous()
    mm = mm_224.contiguous()
    N, C = mm.shape
    assert target.numel() == N

    # Output is transposed: [C, N], bf16
    out = torch.empty((C, N), dtype=torch.bfloat16, device=mm.device)

    inv_total_weight = 1.0 / 8192.0
    ignore_index = -100

    BLOCK_C = 4096
    grid = (N,)
    _fused_kernel[grid](
        mm, target, out,
        N, C,
        inv_total_weight,
        ignore_index,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )
    return out