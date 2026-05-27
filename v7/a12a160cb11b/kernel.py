import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_backward_kernel(
    grad_output_ptr,  # bf16 [N, D]
    indices_ptr,      # int64 [N]
    grad_weight_ptr,  # fp32 [num_weights, D]
    N,                # number of indices
    D,                # embedding dim
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
):
    """
    Fused embedding_dense_backward + cast-to-fp32.
    Each program handles one (index, d-block) pair.
    Uses atomic_add to accumulate gradients into the weight row.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_n >= N:
        return

    idx = tl.load(indices_ptr + pid_n)

    # Skip padding_idx
    if idx == padding_idx:
        return

    # Bounds check
    if (idx < 0) | (idx >= num_weights):
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D

    grad_vals = tl.load(grad_output_ptr + pid_n * D + offs_d, mask=mask, other=0.0)
    grad_vals_f32 = grad_vals.to(tl.float32)

    tl.atomic_add(grad_weight_ptr + idx * D + offs_d, grad_vals_f32, mask=mask)


def kernel_function(add_223, arg583_1):
    """
    Fused embedding_dense_backward -> _to_copy(fp32).
    
    Fusion: We compute the embedding backward (scatter-add of grad_output rows
    into a [num_weights, D] gradient tensor) directly in fp32, fusing the
    subsequent _to_copy(fp32) cast into the same kernel by accumulating in fp32.
    
    Args:
        add_223: grad_output tensor [..., D] in bf16
        arg583_1: indices tensor [...] in int64
    
    Returns:
        grad_weight tensor [num_weights, D] in fp32
    """
    num_weights = 128256
    padding_idx = -1

    assert add_223.is_cuda and arg583_1.is_cuda
    assert add_223.dtype == torch.bfloat16
    assert arg583_1.dtype == torch.int64

    # Flatten
    D = add_223.shape[-1]
    N = arg583_1.numel()
    grad_output_flat = add_223.reshape(N, D).contiguous()
    indices_flat = arg583_1.reshape(N).contiguous()

    # Allocate output in fp32 (this fuses the _to_copy cast)
    grad_weight = torch.zeros((num_weights, D), dtype=torch.float32, device=add_223.device)

    BLOCK_D = 256
    grid = (N, triton.cdiv(D, BLOCK_D))

    _embedding_backward_kernel[grid](
        grad_output_flat,
        indices_flat,
        grad_weight,
        N,
        D,
        num_weights,
        padding_idx,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return grad_weight