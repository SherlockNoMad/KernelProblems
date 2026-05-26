import torch
import triton
import triton.language as tl

# Fused kernel:
# - Reinterpret wait_tensor_871 (bf16, 525336576 elements) as [8, 65667072] view
# - Take split [0] -> first 65667072 elements per row... actually total = 8*65667072 = 525336576
# - View as bf16 of shape [128256, 4096] (reinterpret on the slice)
# - Embedding lookup with arg583_1 indices of shape [1, 8192]
# Output: [1, 8192, 4096] bf16
#
# Since wait_tensor is already bf16 and total size matches 128256*4096*8/8 = ... let's check:
# 128256 * 4096 = 525336576 = total bf16 elements. The split takes [65667072] columns of int8 view?
# Actually view_default reshapes to [8, -1] -> [8, 65667072]. Then split on dim=1 with [65667072]
# gives the whole thing (only one chunk). view_dtype to bf16 keeps it bf16 (already bf16).
# Then view to [128256, 4096].
# So effectively: embedding(wait_tensor_871.view(128256, 4096), arg583_1)

@triton.jit
def _embedding_kernel(
    weight_ptr,  # bf16 [V, D]
    idx_ptr,     # int64 [N]
    out_ptr,     # bf16 [N, D]
    N, D,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    idx = tl.load(idx_ptr + pid_n).to(tl.int64)
    
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D
    
    w = tl.load(weight_ptr + idx * D + offs_d, mask=mask, other=0.0)
    tl.store(out_ptr + pid_n * D + offs_d, w, mask=mask)


def kernel_function(wait_tensor_871, arg583_1):
    """
    Fused embedding lookup from a flat bf16 buffer reinterpreted as [128256, 4096].
    """
    assert wait_tensor_871.is_cuda and arg583_1.is_cuda
    assert wait_tensor_871.dtype == torch.bfloat16
    assert wait_tensor_871.numel() == 525336576
    
    V = 128256
    D = 4096
    
    # Reinterpret as [V, D] - just a view, no compute
    weight = wait_tensor_871.view(V, D)
    
    idx_shape = arg583_1.shape  # [1, 8192]
    idx_flat = arg583_1.contiguous().view(-1)
    N = idx_flat.numel()
    
    out_shape = list(idx_shape) + [D]
    output = torch.empty(out_shape, dtype=torch.bfloat16, device=wait_tensor_871.device)
    out_flat = output.view(N, D)
    
    BLOCK_D = 256
    grid = (N, triton.cdiv(D, BLOCK_D))
    
    _embedding_kernel[grid](
        weight, idx_flat, out_flat,
        N, D,
        BLOCK_D=BLOCK_D,
    )
    
    return output