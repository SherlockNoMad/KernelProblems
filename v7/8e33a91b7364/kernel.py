import torch
import triton
import triton.language as tl


# Fused kernel pipeline:
# 1. Extract weight from wait_tensor_969: bytes [0:512*2] -> view as bf16 [256] ... 
#    Wait: split_with_sizes on view [8, -1] with [512, 65667072] along dim=1.
#    view_default shape: [8, 65667584] (since 525340672/8 = 65667584).
#    getitem[0]: [8, 512] -> view_dtype bf16 (same dtype) -> clone -> _unsafe_view [4096]
#    So weight = wait_tensor_969 viewed as [8, 65667584], take [:, :512], flatten -> 4096 elements
# 2. add_tensor = add_62 + mm_223 (view as [1,8192,4096])
# 3. RMS norm along last dim 4096 with weight, eps=1e-5 -> out1 shape [8192, 4096]
# 4. getitem[1]: [8, 65667072] -> view bf16 -> clone -> _unsafe_view [128256, 4096]
#    8 * 65667072 = 525336576 elements = 128256 * 4096. Correct.
#    out2 is just a reshape/copy of wait_tensor_969[:, 512:] flattened.


@triton.jit
def _rmsnorm_fused_kernel(
    add62_ptr,      # [N, D] bf16
    mm_ptr,         # [N, D] bf16
    weight_ptr,     # [D] bf16
    out_ptr,        # [N, D] bf16
    N, D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    
    row_off = row * D
    
    a = tl.load(add62_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b
    
    # Compute mean of squares
    sq = x * x
    sq = tl.where(mask, sq, 0.0)
    mean_sq = tl.sum(sq, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    y = x * rstd * w
    
    tl.store(out_ptr + row_off + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _extract_weight_kernel(
    wait_ptr,       # bf16 flat [525340672]
    weight_ptr,     # bf16 [4096]
    stride_row,     # 65667584
    n_per_row,      # 512
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # weight[i] for i in [0, 4096): which row? i // 512, col i % 512
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL
    
    row = offs // n_per_row
    col = offs % n_per_row
    src_idx = row * stride_row + col
    
    v = tl.load(wait_ptr + src_idx, mask=mask, other=0.0)
    tl.store(weight_ptr + offs, v, mask=mask)


@triton.jit
def _extract_out2_kernel(
    wait_ptr,       # bf16 flat
    out_ptr,        # bf16 flat [128256*4096]
    stride_row,     # 65667584
    skip,           # 512
    n_per_row,      # 65667072
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL
    
    row = offs // n_per_row
    col = offs % n_per_row
    src_idx = row * stride_row + skip + col
    
    v = tl.load(wait_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v, mask=mask)


def kernel_function(mm_223, add_62, wait_tensor_969):
    """
    Fused kernel implementing:
      - Extract weight (4096) from wait_tensor split[0]
      - Add mm_223 + add_62, then RMSNorm with weight -> out1 [8192, 4096]
      - Extract & reshape split[1] -> out2 [128256, 4096]
    """
    assert mm_223.is_cuda and add_62.is_cuda and wait_tensor_969.is_cuda
    assert mm_223.dtype == torch.bfloat16
    assert add_62.dtype == torch.bfloat16
    assert wait_tensor_969.dtype == torch.bfloat16
    
    device = mm_223.device
    
    # View parameters
    # wait_tensor_969: [525340672] flat, viewed as [8, 65667584]
    stride_row = 65667584
    assert wait_tensor_969.numel() == 8 * stride_row
    
    # Extract weight [4096]
    weight = torch.empty(4096, dtype=torch.bfloat16, device=device)
    BLOCK = 256
    grid_w = (triton.cdiv(4096, BLOCK),)
    _extract_weight_kernel[grid_w](
        wait_tensor_969, weight,
        stride_row, 512,
        TOTAL=4096, BLOCK_SIZE=BLOCK,
    )
    
    # RMSNorm fused with add
    N = 8192
    D = 4096
    out1 = torch.empty((N, D), dtype=torch.bfloat16, device=device)
    
    # Ensure contiguous flat views
    add62_flat = add_62.contiguous().view(N, D)
    mm_flat = mm_223.contiguous().view(N, D)
    
    grid_rms = (N,)
    _rmsnorm_fused_kernel[grid_rms](
        add62_flat, mm_flat, weight, out1,
        N, D, 1e-5,
        BLOCK_SIZE=4096,
    )
    
    # Extract out2 [128256, 4096] = 525336576 elements
    out2_total = 128256 * 4096
    out2 = torch.empty((128256, 4096), dtype=torch.bfloat16, device=device)
    
    BLOCK2 = 1024
    grid_o2 = (triton.cdiv(out2_total, BLOCK2),)
    _extract_out2_kernel[grid_o2](
        wait_tensor_969, out2,
        stride_row, 512, 65667072,
        out2_total, BLOCK_SIZE=BLOCK2,
    )
    
    return (out1, out2)