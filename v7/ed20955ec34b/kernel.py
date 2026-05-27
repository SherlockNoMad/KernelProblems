import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_bwd_fused_kernel(
    mm672_ptr, view1794_ptr, mm674_ptr, emb_ptr, rstd_ptr, w_ptr, add220_ptr,
    out_ptr, gw_partial_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_off = row * N

    # Match reference: bf16 additions
    g1 = tl.load(view1794_ptr + row_off + cols, mask=mask, other=0.0)
    g2 = tl.load(mm672_ptr + row_off + cols, mask=mask, other=0.0)
    g3 = tl.load(mm674_ptr + row_off + cols, mask=mask, other=0.0)
    # add_tensor = view_1794 + view(mm_672)  -- bf16
    s1 = (g1.to(tl.float32) + g2.to(tl.float32)).to(tl.bfloat16)
    # add_tensor_1 = add_tensor + view(mm_674) -- bf16
    grad_y_bf16 = (s1.to(tl.float32) + g3.to(tl.float32)).to(tl.bfloat16)
    grad_y = grad_y_bf16.to(tl.float32)

    x = tl.load(emb_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    xhat = x * rstd
    g = grad_y * w

    # Store grad_w partial (per row): grad_y * xhat (still need to sum across rows)
    gw_partial = grad_y * xhat
    tl.store(gw_partial_ptr + row_off + cols, gw_partial, mask=mask)

    # c = mean(g * xhat)
    c = tl.sum(g * xhat, axis=0) / N

    grad_x = rstd * (g - xhat * c)

    add220 = tl.load(add220_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    out = add220 + grad_x

    tl.store(out_ptr + row_off + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _grad_w_reduce_kernel(
    gw_partial_ptr, gw_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r_start in range(0, M, BLOCK_M):
        rows = r_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        # Load block [BLOCK_M, BLOCK_N]
        offs = rows[:, None] * N + cols[None, :]
        m = row_mask[:, None] & col_mask[None, :]
        v = tl.load(gw_partial_ptr + offs, mask=m, other=0.0)
        acc += tl.sum(v, axis=0)

    tl.store(gw_ptr + cols, acc, mask=col_mask)


def kernel_function(mm_672, view_1794, mm_674, embedding, getitem_1_recomputed, _unsafe_view_432, add_220):
    assert mm_672.is_cuda
    M = 8192
    N = 4096

    mm672_f = mm_672.contiguous().view(M, N)
    view1794_f = view_1794.contiguous().view(M, N)
    mm674_f = mm_674.contiguous().view(M, N)
    emb_f = embedding.contiguous().view(M, N)
    add220_f = add_220.contiguous().view(M, N)
    rstd_f = getitem_1_recomputed.contiguous().view(M)
    w_f = _unsafe_view_432.contiguous()

    out0 = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_672.device)
    out0_f = out0.view(M, N)

    gw_partial = torch.empty((M, N), dtype=torch.float32, device=mm_672.device)

    _rmsnorm_bwd_fused_kernel[(M,)](
        mm672_f, view1794_f, mm674_f, emb_f, rstd_f, w_f, add220_f,
        out0_f, gw_partial,
        M, N,
        BLOCK_N=4096,
        num_warps=8,
    )

    grad_w = torch.empty((N,), dtype=torch.float32, device=mm_672.device)
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _grad_w_reduce_kernel[grid](
        gw_partial, grad_w,
        M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return (out0, grad_w)