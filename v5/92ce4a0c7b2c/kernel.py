import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    N: tl.constexpr, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    
    tl.store(rstd_ptr + row, rstd)
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    mm1_ptr, mm2_ptr, mm3_ptr,
    x_ptr, w_ptr, rstd_ptr,
    add215_ptr,
    out_grad_x_ptr,
    gyx_ptr,
    N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # Match reference: add.Tensor between bf16 tensors stores intermediate as bf16
    gy1 = tl.load(mm1_ptr + row * N + cols, mask=mask, other=0.0)
    gy2 = tl.load(mm2_ptr + row * N + cols, mask=mask, other=0.0)
    gy3 = tl.load(mm3_ptr + row * N + cols, mask=mask, other=0.0)
    # First addition: gy1 + gy2, result cast back to bf16
    gy12 = (gy1.to(tl.float32) + gy2.to(tl.float32)).to(tl.bfloat16)
    # Second addition: gy12 + gy3, result cast back to bf16
    gy_bf16 = (gy12.to(tl.float32) + gy3.to(tl.float32)).to(tl.bfloat16)
    gy = gy_bf16.to(tl.float32)
    
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    
    c1 = gy * w
    xhat = x * rstd
    mean_term = tl.sum(c1 * xhat, axis=0) / N
    grad_x = rstd * (c1 - xhat * mean_term)
    
    add215 = tl.load(add215_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    out = add215 + grad_x
    tl.store(out_grad_x_ptr + row * N + cols, out.to(out_grad_x_ptr.dtype.element_ty), mask=mask)
    
    gyx = gy * xhat
    tl.store(gyx_ptr + row * N + cols, gyx, mask=mask)


@triton.jit
def _grad_w_reduce_kernel(
    gyx_ptr, gw_ptr,
    M, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    mask_c = cols < N
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        mask_r = rows < M
        ptrs = gyx_ptr + rows[:, None] * N + cols[None, :]
        m = mask_r[:, None] & mask_c[None, :]
        vals = tl.load(ptrs, mask=m, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    tl.store(gw_ptr + cols, acc, mask=mask_c)


def kernel_function(getitem_818, add_1, mm_656, mm_658, mm_660, add_215):
    assert add_1.shape == (1, 8192, 4096)
    
    weight = getitem_818.contiguous().view(-1)
    
    M = 8192
    N = 4096
    eps = 1e-5
    
    x_flat = add_1.view(M, N)
    add215_flat = add_215.view(M, N)
    
    y = torch.empty((M, N), dtype=torch.bfloat16, device=add_1.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=add_1.device)
    
    BLOCK_N = 4096
    _rmsnorm_fwd_kernel[(M,)](
        x_flat, weight, y, rstd,
        N=N, eps=eps, BLOCK_N=BLOCK_N,
    )
    
    out_grad_x = torch.empty((1, M, N), dtype=torch.bfloat16, device=add_1.device)
    gyx_temp = torch.empty((M, N), dtype=torch.float32, device=add_1.device)
    
    _rmsnorm_bwd_kernel[(M,)](
        mm_656, mm_658, mm_660,
        x_flat, weight, rstd,
        add215_flat,
        out_grad_x.view(M, N), gyx_temp,
        N=N, BLOCK_N=BLOCK_N,
    )
    
    grad_w = torch.empty((N,), dtype=torch.float32, device=add_1.device)
    BLOCK_M = 64
    BLOCK_N_R = 128
    grid_r = (triton.cdiv(N, BLOCK_N_R),)
    _grad_w_reduce_kernel[grid_r](
        gyx_temp, grad_w,
        M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_R,
    )
    
    return (y, y, y, out_grad_x, grad_w)