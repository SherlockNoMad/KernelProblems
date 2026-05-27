import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(
    weight_ptr,  # bf16 view of wait_tensor_871 first portion, shape [V, H]
    idx_ptr,     # int64 indices, shape [N]
    out_ptr,     # bf16 output, shape [N, H]
    N,           # number of indices
    H: tl.constexpr,  # embedding dim (4096)
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_n >= N:
        return

    idx = tl.load(idx_ptr + pid_n)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    w = tl.load(weight_ptr + idx * H + offs_h, mask=mask_h, other=0.0)
    tl.store(out_ptr + pid_n * H + offs_h, w, mask=mask_h)


def kernel_function(wait_tensor_871, arg583_1):
    """
    Fused operation:
      1. View wait_tensor_871 as [8, -1], split off first 65667072 elements
      2. Reinterpret as bf16, view as [128256, 4096] embedding weight
      3. Perform embedding lookup with indices arg583_1
    
    Since wait_tensor_871 is already bf16 in the test, the reinterpret is a no-op view.
    We treat the first 128256*4096 bf16 elements as the embedding table.
    """
    assert wait_tensor_871.is_cuda
    assert arg583_1.is_cuda

    V = 128256
    H = 4096
    weight_numel = V * H  # 525336576

    # The original computation: view as [8, N/8], take first 65667072 cols (bf16 elements
    # since input is already bf16 in test). With 8 rows and 65667072 cols => 8*65667072 = 525336576
    # bf16 elements, then viewed as [128256, 4096].
    # So effectively the first weight_numel bf16 elements form the weight in row-major order.
    
    # Get a bf16 contiguous view of the weight portion
    flat_bf16 = wait_tensor_871.view(torch.bfloat16) if wait_tensor_871.dtype != torch.bfloat16 else wait_tensor_871
    
    # Per the model: view [8, -1], split [65667072] on dim 1 -> [8, 65667072]
    # Then view as bf16 (no-op here), then view [128256, 4096]
    # The split takes contiguous elements: rows 0..7 each contribute 65667072 elements,
    # but split_with_sizes on dim=1 with [65667072] gives shape [8, 65667072] which when
    # flattened in C-order is NOT the same as the first 525336576 elements of the original.
    # Wait: view_default has shape [8, N/8]. Split on dim 1 takes first 65667072 cols.
    # Since N/8 = 525336576/8... let me check: 525336576/8 = 65667072. So split takes ALL of it!
    # So getitem == view_default, and total = 525336576 bf16 elements.
    
    weight = flat_bf16[:weight_numel].view(V, H)

    indices = arg583_1.contiguous().view(-1)
    N = indices.numel()
    out_shape = list(arg583_1.shape) + [H]

    output = torch.empty(out_shape, dtype=torch.bfloat16, device=wait_tensor_871.device)
    out_flat = output.view(N, H)

    BLOCK_H = 512
    grid = (N, triton.cdiv(H, BLOCK_H))
    _embedding_kernel[grid](
        weight, indices, out_flat,
        N, H=H, BLOCK_H=BLOCK_H,
    )

    return output