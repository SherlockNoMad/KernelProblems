import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,
    idx_ptr,
    out_ptr,
    N,
    TOKENS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * TOKENS_PER_BLOCK
    
    offs_t = base + tl.arange(0, TOKENS_PER_BLOCK)
    mask_t = offs_t < N
    idx = tl.load(idx_ptr + offs_t, mask=mask_t, other=0).to(tl.int64)
    
    offs_d = tl.arange(0, D)
    src = weight_ptr + idx[:, None] * D + offs_d[None, :]
    dst = out_ptr + offs_t[:, None] * D + offs_d[None, :]
    
    w = tl.load(src, mask=mask_t[:, None])
    tl.store(dst, w, mask=mask_t[:, None])


def kernel_function(wait_tensor_871, arg583_1):
    V = 128256
    D = 4096
    
    weight = wait_tensor_871.view(V, D)
    
    idx_shape = arg583_1.shape
    idx_flat = arg583_1.contiguous().view(-1)
    N = idx_flat.numel()
    
    out_shape = list(idx_shape) + [D]
    output = torch.empty(out_shape, dtype=torch.bfloat16, device=wait_tensor_871.device)
    out_flat = output.view(N, D)
    
    TOKENS_PER_BLOCK = 8
    grid = (triton.cdiv(N, TOKENS_PER_BLOCK),)
    
    _embedding_kernel[grid](
        weight, idx_flat, out_flat,
        N,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        D=D,
        num_warps=16,
        num_stages=2,
    )
    
    return output