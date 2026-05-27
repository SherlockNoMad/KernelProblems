import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + row * N + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_bwd_dx_kernel(
    g1_ptr, g2_ptr, g3_ptr,
    x_ptr, w_ptr, rstd_ptr,
    add215_ptr, out_ptr,
    dweight_partial_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Match reference: add in bf16 then convert (mimics view+add+add chain)
    g1_bf = tl.load(g1_ptr + row * N + offs, mask=mask, other=0.0)
    g2_bf = tl.load(g2_ptr + row * N + offs, mask=mask, other=0.0)
    g3_bf = tl.load(g3_ptr + row * N + offs, mask=mask, other=0.0)
    # First add: g1 + g2 in bf16
    s1_bf = (g1_bf.to(tl.float32) + g2_bf.to(tl.float32)).to(tl.bfloat16)
    # Second add: s1 + g3 in bf16
    s2_bf = (s1_bf.to(tl.float32) + g3_bf.to(tl.float32)).to(tl.bfloat16)
    dy = s2_bf.to(tl.float32)

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dw_partial = dy * x_hat
    tl.store(dweight_partial_ptr + row * N + offs, dw_partial, mask=mask)

    dy_w = dy * w
    c = tl.sum(dy_w * x_hat, axis=0) / N
    dx = rstd * (dy_w - x_hat * c)

    add215 = tl.load(add215_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    result = add215 + dx
    tl.store(out_ptr + row * N + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _reduce_dweight_kernel(
    dw_partial_ptr, dw_out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m < M
        vals = tl.load(dw_partial_ptr + offs_m * N + col, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(dw_out_ptr + col, acc)


def kernel_function(getitem_818, add_1, mm_656, mm_658, mm_660, add_215):
    assert add_1.is_cuda and add_1.dtype == torch.bfloat16
    
    weight_view = getitem_818.contiguous().view(-1)
    assert weight_view.numel() == 4096
    
    M = 8192
    N = 4096
    eps = 1e-5
    
    x = add_1.view(M, N)
    
    y = torch.empty((M, N), dtype=torch.bfloat16, device=add_1.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=add_1.device)
    
    BLOCK_N = 4096
    _rms_fwd_kernel[(M,)](
        x, weight_view, y, rstd,
        N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    
    y_view0 = y
    y_view1 = y
    y_view2 = y
    
    add_215_2d = add_215.view(M, N)
    out_add = torch.empty((1, 8192, 4096), dtype=torch.bfloat16, device=add_1.device)
    out_add_2d = out_add.view(M, N)
    
    dw_partial = torch.empty((M, N), dtype=torch.float32, device=add_1.device)
    
    _rms_bwd_dx_kernel[(M,)](
        mm_656, mm_658, mm_660,
        x, weight_view, rstd,
        add_215_2d, out_add_2d,
        dw_partial,
        M, N,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    
    dw = torch.empty((N,), dtype=torch.float32, device=add_1.device)
    BLOCK_M_RED = 256
    _reduce_dweight_kernel[(N,)](
        dw_partial, dw,
        M, N,
        BLOCK_M=BLOCK_M_RED,
        num_warps=4,
    )
    
    return (y_view0, y_view1, y_view2, out_add, dw)