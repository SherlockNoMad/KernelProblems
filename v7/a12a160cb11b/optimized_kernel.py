import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    N,
    D,
    num_weights,
    padding_idx,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)

    d_start = pid_d * BLOCK_D
    offs_d = d_start + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    n_start = pid_n * BLOCK_N

    for i in tl.static_range(BLOCK_N):
        n = n_start + i
        if n < N:
            idx = tl.load(indices_ptr + n)
            if (idx != padding_idx) & (idx >= 0) & (idx < num_weights):
                grad_vals = tl.load(grad_output_ptr + n * D + offs_d, mask=mask_d, other=0.0)
                grad_vals_f32 = grad_vals.to(tl.float32)
                tl.atomic_add(grad_weight_ptr + idx * D + offs_d, grad_vals_f32, mask=mask_d)


def kernel_function(add_223, arg583_1):
    num_weights = 128256
    padding_idx = -1

    D = add_223.shape[-1]
    N = arg583_1.numel()
    grad_output_flat = add_223.reshape(N, D)
    indices_flat = arg583_1.reshape(N)

    grad_weight = torch.zeros((num_weights, D), dtype=torch.float32, device=add_223.device)

    BLOCK_D = 4096
    BLOCK_N = 8
    grid = (triton.cdiv(D, BLOCK_D), triton.cdiv(N, BLOCK_N))

    _embedding_backward_kernel[grid](
        grad_output_flat,
        indices_flat,
        grad_weight,
        N,
        D,
        num_weights,
        padding_idx,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=16,
        num_stages=2,
    )

    return grad_weight