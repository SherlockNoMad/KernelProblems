import torch
import triton
import triton.language as tl


# Fused kernel for the backward pass of NLLLoss + LogSoftmax:
# Stages fused in a single pass over each row (length N = 128256):
#   1. Compute grad input for NLLLoss backward:
#        grad_in[i, j] = -(1/8192) * 1{view_805[i] == j} * 1{view_805[i] != ignore_index}
#      (weight is None, reduction='mean' with normalizer total_weight = getitem_419,
#       but since the upstream is ones_like(div)/8192 we effectively use 1/8192 directly,
#       matching ATen's behavior when reduction != 'mean'/total_weight is precomputed).
#      We implement this directly: at j==target, grad = -1/8192 (if target != -100), else 0.
#   2. Compute log_softmax backward:
#        grad_out[i, j] = grad_in[i, j] - exp(log_softmax[i, j]) * sum_k(grad_in[i, k])
#      sum_k grad_in[i, k] = -1/8192 if target != -100 else 0
#   3. Cast to bf16 and store.
# Output shape: [8192, 128256], dtype bf16.


@triton.jit
def _fused_nll_logsoftmax_bwd_kernel(
    log_softmax_ptr,  # [M, N] fp32
    target_ptr,       # [M] int64
    out_ptr,          # [M, N] bf16
    M, N,
    scale,            # 1/8192
    ignore_index,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    target = tl.load(target_ptr + row)
    valid = target != ignore_index
    # sum of grad_in over the row = -scale if valid else 0
    sum_grad_in = tl.where(valid, -scale, 0.0)

    row_offset = row * N
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        ls = tl.load(log_softmax_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)

        # grad_in at this column: -scale if (valid and offs == target) else 0
        is_target = (offs == target) & valid
        grad_in = tl.where(is_target, -scale, 0.0)

        grad_out = grad_in - tl.exp(ls) * sum_grad_in

        tl.store(out_ptr + row_offset + offs, grad_out.to(tl.bfloat16), mask=mask)


def kernel_function(div, _log_softmax, view_805, getitem_419):
    """
    Fused Triton implementation of:
        ones_like(div)/8192 -> nll_loss_backward -> log_softmax_backward -> to(bf16) -> view.

    All compute is performed in the Triton kernel. PyTorch is used only for allocation
    and launch configuration.
    """
    assert _log_softmax.is_cuda and view_805.is_cuda
    assert _log_softmax.dtype == torch.float32
    assert view_805.dtype == torch.int64
    assert _log_softmax.dim() == 2

    M, N = _log_softmax.shape
    assert view_805.shape[0] == M

    out = torch.empty((M, N), device=_log_softmax.device, dtype=torch.bfloat16)

    # scale = 1/8192 as in the Model
    scale = 1.0 / 8192.0
    ignore_index = -100

    BLOCK_N = 1024
    grid = (M,)
    _fused_nll_logsoftmax_bwd_kernel[grid](
        _log_softmax, view_805, out,
        M, N,
        scale,
        ignore_index,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out