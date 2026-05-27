import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    idx = tl.load(idx_ptr + offs_n, mask=mask_n, other=0)
    src = weight_ptr + idx[:, None] * H + offs_h[None, :]
    dst = out_ptr + offs_n[:, None] * H + offs_h[None, :]
    w = tl.load(src, mask=mask_n[:, None])
    tl.store(dst, w, mask=mask_n[:, None])


def kernel_function(wait_tensor_871, arg583_1):
    V = 128256
    H = 4096
    weight_numel = V * H

    flat_bf16 = wait_tensor_871 if wait_tensor_871.dtype == torch.bfloat16 else wait_tensor_871.view(torch.bfloat16)
    weight = flat_bf16[:weight_numel].view(V, H)

    indices = arg583_1.contiguous().view(-1)
    N = indices.numel()
    out_shape = list(arg583_1.shape) + [H]

    output = torch.empty(out_shape, dtype=torch.bfloat16, device=wait_tensor_871.device)
    out_flat = output.view(N, H)

    BLOCK_N = 8
    BLOCK_H = 512
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))
    _embedding_kernel[grid](
        weight, indices, out_flat,
        N, H=H, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=4,
    )

    return output