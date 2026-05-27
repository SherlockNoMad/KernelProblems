import torch
import triton
import triton.language as tl


# Fused operations in this implementation:
# 1. Add: add_62 + mm_223 (viewed as [1, 8192, 4096])
# 2. RMSNorm on the result of (1) along last dim (4096), weighted by first 512*8=4096 bf16 elements
#    of wait_tensor_969 (reinterpreted from 512 elements of view [8, ...] -> first 512 cols).
#    Actually: view_default = wait_tensor_969.view(8, -1); split [512, 65667072] along dim=1.
#    getitem (8, 512) -> view as bf16 -> contiguous -> view(4096) -> RMS weight.
# 3. Output1: rms_norm(add_tensor, weight) reshaped to (8192, 4096)
# 4. Output2: second split (8, 65667072) -> bf16 -> contiguous -> view(128256, 4096)


@triton.jit
def _add_rmsnorm_kernel(
    add_ptr,        # add_62, [8192, 4096] (1*8192*4096 contiguous)
    mm_ptr,         # mm_223, [8192, 4096]
    w_ptr,          # rms weight, [4096]
    out_ptr,        # output [8192, 4096]
    N,              # 4096
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_off = row * N
    a = tl.load(add_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mm_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    x = a + m

    # RMS norm
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_ptr + row_off + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


def kernel_function(mm_223, add_62, wait_tensor_969):
    """
    Fused implementation:
      - out1 = RMSNorm(add_62 + mm_223.view(1,8192,4096), weight=wait_tensor_969[:4096_bf16])
              reshaped to [8192, 4096]
      - out2 = wait_tensor_969 split second part viewed as bf16 reshaped to [128256, 4096]
    """
    assert mm_223.is_cuda and add_62.is_cuda and wait_tensor_969.is_cuda
    assert mm_223.dtype == torch.bfloat16
    assert add_62.dtype == torch.bfloat16
    assert wait_tensor_969.dtype == torch.bfloat16

    # Reproduce: wait_tensor_969.view(8, -1) then split [512, 65667072] along dim=1
    # The wait tensor total = 8 * (512 + 65667072) = 8 * 65667584 = 525340672 ✓
    # First part (8, 512) bf16, then viewed as bf16 (same), contiguous, viewed as [4096]
    # Since wait_tensor_969 is already bf16, view.dtype to bf16 is a no-op.
    # We need contiguous of the slice [:, :512] of shape (8, 65667584).

    total_per_row = 65667584  # 512 + 65667072
    assert wait_tensor_969.numel() == 8 * total_per_row

    wt_view = wait_tensor_969.view(8, total_per_row)

    # First slice: (8, 512) -> contiguous -> view(4096)
    first_slice = wt_view[:, :512]  # non-contiguous strided view
    rms_weight = torch.empty(4096, dtype=torch.bfloat16, device=wait_tensor_969.device)
    # Copy using triton kernel. But we need it contiguous. Use a simple gather kernel.
    # Easiest: use .contiguous() which is a memory copy (allowed as alloc/copy, not compute).
    # To strictly avoid pytorch compute ops, do manual copy via triton.
    # We'll do row-by-row copy using a triton kernel that knows strides.

    # Use a triton kernel to copy strided -> contiguous
    @triton.jit
    def _strided_copy_kernel(
        src_ptr, dst_ptr,
        stride0, stride1,
        D0, D1,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        # each pid handles one row chunk
        row = pid // tl.cdiv(D1, BLOCK)
        col_blk = pid % tl.cdiv(D1, BLOCK)
        cols = col_blk * BLOCK + tl.arange(0, BLOCK)
        mask = (cols < D1) & (row < D0)
        src_off = row * stride0 + cols * stride1
        dst_off = row * D1 + cols
        x = tl.load(src_ptr + src_off, mask=mask)
        tl.store(dst_ptr + dst_off, x, mask=mask)

    # Copy first_slice (8, 512) to contiguous rms_weight viewed as (8, 512)
    rms_weight_2d = rms_weight.view(8, 512)
    D0, D1 = 8, 512
    BLOCK = 512
    grid = (D0 * triton.cdiv(D1, BLOCK),)
    _strided_copy_kernel[grid](
        first_slice, rms_weight_2d,
        first_slice.stride(0), first_slice.stride(1),
        D0, D1, BLOCK=BLOCK,
    )

    # Second slice: (8, 65667072) -> contiguous -> view(128256, 4096)
    second_slice = wt_view[:, 512:]  # (8, 65667072)
    out2 = torch.empty(128256, 4096, dtype=torch.bfloat16, device=wait_tensor_969.device)
    # 128256 * 4096 = 525340672 - 4096 = ... let's check: 8*65667072 = 525336576; 128256*4096 = 525336576 ✓
    out2_flat = out2.view(8, 65667072)
    D0, D1 = 8, 65667072
    BLOCK = 4096
    grid = (D0 * triton.cdiv(D1, BLOCK),)
    _strided_copy_kernel[grid](
        second_slice, out2_flat,
        second_slice.stride(0), second_slice.stride(1),
        D0, D1, BLOCK=BLOCK,
    )

    # Now do the fused add + RMSNorm
    # add_62 shape [1, 8192, 4096], mm_223 [8192, 4096]
    M, N = 8192, 4096
    out1 = torch.empty(M, N, dtype=torch.bfloat16, device=mm_223.device)

    add_flat = add_62.view(M, N)
    mm_flat = mm_223.view(M, N)
    assert add_flat.is_contiguous()
    assert mm_flat.is_contiguous()

    BLOCK_SIZE = 4096  # N is 4096, fits
    grid = (M,)
    _add_rmsnorm_kernel[grid](
        add_flat, mm_flat, rms_weight, out1,
        N, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out1, out2