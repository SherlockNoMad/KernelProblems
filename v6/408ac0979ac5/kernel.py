import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    mm3_ptr, emb_ptr, w_ptr,
    add_tensor_ptr, y_ptr, rstd_ptr,
    eps, N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    
    mm3 = tl.load(mm3_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    emb = tl.load(emb_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x_f32 = mm3 + emb
    x_bf = x_f32.to(add_tensor_ptr.dtype.element_ty)
    tl.store(add_tensor_ptr + row * N + cols, x_bf, mask=mask)
    # Reload from bf16 to match what reference sees
    x = x_bf.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(rstd_ptr + row, rstd)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_dx_kernel(
    mm664_ptr, mm666_ptr,
    add_tensor_ptr, w_ptr, rstd_ptr,
    add218_ptr, out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    
    dy1 = tl.load(mm664_ptr + row * N + cols, mask=mask, other=0.0)
    dy2 = tl.load(mm666_ptr + row * N + cols, mask=mask, other=0.0)
    # Add in bf16 to match reference (add_tensor_1 is bf16)
    dy_bf = dy1 + dy2
    dy = dy_bf.to(tl.float32)
    x = tl.load(add_tensor_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    
    x_hat = x * rstd
    dy_w = dy * w
    c = tl.sum(dy_w * x_hat, axis=0) / N
    dx = rstd * (dy_w - x_hat * c)
    
    add218 = tl.load(add218_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    out = add218 + dx
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_dw_kernel(
    mm664_ptr, mm666_ptr,
    add_tensor_ptr, rstd_ptr,
    dw_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for m_start in range(0, M, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        
        offs = rows[:, None] * N + cols[None, :]
        mm = row_mask[:, None] & col_mask[None, :]
        dy1 = tl.load(mm664_ptr + offs, mask=mm, other=0.0)
        dy2 = tl.load(mm666_ptr + offs, mask=mm, other=0.0)
        # Add in bf16 then cast - matches reference precision
        dy_bf = dy1 + dy2
        dy = dy_bf.to(tl.float32)
        x = tl.load(add_tensor_ptr + offs, mask=mm, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd[:, None]
        acc += tl.sum(dy * x_hat, axis=0)
    
    tl.store(dw_ptr + cols, acc, mask=col_mask)


def kernel_function(mm_3, embedding, _unsafe_view_428, mm_664, mm_666, add_218):
    """
    Fused RMS norm forward and backward.
    Stages fused per kernel:
    - Forward: add(mm_3, embedding) -> rms_norm forward (writes add_tensor, y, rstd)
    - Backward dx: add(mm_664, mm_666) [bf16] -> rms_norm_bwd dx -> add(add_218) [fused]
    - Backward dw: reduction across rows (kept separate due to row-reduction pattern)
    """
    assert mm_3.is_cuda
    M, N = 8192, 4096
    eps = 1e-5
    
    device = mm_3.device
    bf16 = torch.bfloat16
    
    add_tensor = torch.empty((1, M, N), device=device, dtype=bf16)
    y = torch.empty((M, N), device=device, dtype=bf16)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    
    BLOCK = 4096
    
    _fwd_kernel[(M,)](
        mm_3, embedding, _unsafe_view_428,
        add_tensor, y, rstd,
        eps, N, BLOCK,
        num_warps=8,
    )
    
    add_tensor_2 = torch.empty((1, M, N), device=device, dtype=bf16)
    
    _bwd_dx_kernel[(M,)](
        mm_664, mm_666,
        add_tensor, _unsafe_view_428, rstd,
        add_218, add_tensor_2,
        N, BLOCK,
        num_warps=8,
    )
    
    dw = torch.empty((N,), device=device, dtype=torch.float32)
    BLOCK_N = 128
    BLOCK_M = 64
    grid = (triton.cdiv(N, BLOCK_N),)
    _bwd_dw_kernel[grid](
        mm_664, mm_666,
        add_tensor, rstd,
        dw,
        M, N, BLOCK_N, BLOCK_M,
        num_warps=4,
    )
    
    view_default = y
    view_default_1 = y
    
    return (view_default, view_default_1, add_tensor_2, dw)