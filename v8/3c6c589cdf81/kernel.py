import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_rms_fwd_dx_kernel(
    mm220_ptr, add61_ptr, mm223_ptr, mm226_ptr, w_ptr,
    x_out_ptr, rstd_ptr, dx_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    base = row * N
    a = tl.load(mm220_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(add61_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(mm223_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(mm226_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    x = a + b + c
    
    # Store x as bf16
    tl.store(x_out_ptr + base + cols, x.to(tl.bfloat16), mask=mask)
    
    # rms norm forward
    s = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(s + eps)
    tl.store(rstd_ptr + row, rstd)
    
    # backward
    dy_w = dy * w
    sum_dy_w_x = tl.sum(dy_w * x, axis=0)
    
    dx = rstd * (dy_w - x * (rstd * rstd) * sum_dy_w_x / N)
    tl.store(dx_ptr + base + cols, dx.to(tl.bfloat16), mask=mask)


@triton.jit
def _dweight_kernel(
    mm226_ptr, x_ptr, rstd_ptr, dw_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    for m_start in range(0, M, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        # Load rstd [BLOCK_M]
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        # Load dy and x [BLOCK_M, BLOCK_N]
        offs = rows[:, None] * N + col_offsets[None, :]
        mask2d = row_mask[:, None] & col_mask[None, :]
        dy = tl.load(mm226_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask2d, other=0.0).to(tl.float32)
        acc += tl.sum(dy * x * rstd[:, None], axis=0)
    
    tl.store(dw_ptr + col_offsets, acc, mask=col_mask)


def kernel_function(mm_220, add_61, mm_223, mm_226, view_3253):
    # Shapes
    # mm_220: [8192, 4096] bf16
    # add_61: [1, 8192, 4096] bf16
    # mm_223: [8192, 4096] bf16
    # mm_226: [8192, 4096] bf16
    # view_3253: [4096] bf16
    
    M = 8192
    N = 4096
    eps = 1e-5
    
    device = mm_220.device
    
    x_buf = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    rstd_buf = torch.empty((M,), dtype=torch.float32, device=device)
    dx_buf = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    dw_buf = torch.empty((N,), dtype=torch.float32, device=device)
    
    BLOCK_N = triton.next_power_of_2(N)
    
    _fused_add_rms_fwd_dx_kernel[(M,)](
        mm_220, add_61, mm_223, mm_226, view_3253,
        x_buf, rstd_buf, dx_buf,
        M, N, eps,
        BLOCK_N=BLOCK_N,
    )
    
    BLOCK_N2 = 128
    BLOCK_M2 = 64
    grid = (triton.cdiv(N, BLOCK_N2),)
    _dweight_kernel[grid](
        mm_226, x_buf, rstd_buf, dw_buf,
        M, N,
        BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
    )
    
    return dx_buf, dw_buf