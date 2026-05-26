import triton
import triton.language as tl
import torch

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, RSTD_ptr,
    N, D, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    
    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    
    sq = x * x
    mean_sq = tl.sum(sq, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    
    tl.store(RSTD_ptr + row, rstd)
    
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y_ptr + row * D + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _add_bf16_kernel(
    A_ptr, B_ptr, OUT_ptr,
    N_ELEM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    # cast to fp32, add, cast back to bf16 (matches torch bf16 add)
    out = (a.to(tl.float32) + b.to(tl.float32)).to(OUT_ptr.dtype.element_ty)
    tl.store(OUT_ptr + offs, out, mask=mask)


@triton.jit
def _rmsnorm_bwd_dx_kernel(
    GO_ptr, X_ptr, W_ptr, RSTD_ptr,
    GETITEM420_ptr, ADD_OUT_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    
    # Load go (already summed in bf16)
    go = tl.load(GO_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    
    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)
    
    gw = go * w
    sum_gwx = tl.sum(gw * x, axis=0)
    c = sum_gwx * rstd * rstd / D
    
    dx = rstd * (gw - x * c)
    
    g420 = tl.load(GETITEM420_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = g420 + dx
    tl.store(ADD_OUT_ptr + row * D + cols, out.to(ADD_OUT_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_dw_kernel(
    GO_ptr, X_ptr, RSTD_ptr, DW_ptr,
    N, D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * BLOCK_D
    cols = col_start + tl.arange(0, BLOCK_D)
    col_mask = cols < D
    
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for n_start in range(0, N, BLOCK_N):
        rows = n_start + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        
        offs = rows[:, None] * D + cols[None, :]
        m = row_mask[:, None] & col_mask[None, :]
        
        go = tl.load(GO_ptr + offs, mask=m, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        
        contrib = go * x * rstd[:, None]
        acc += tl.sum(contrib, axis=0)
    
    tl.store(DW_ptr + cols, acc, mask=col_mask)


def kernel_function(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420):
    device = add_62_recomputed.device
    
    # Extract weight: getitem_1624 is bf16 [8, 512] strided -> contiguous [4096]
    weight_2d = getitem_1624.contiguous()
    weight = weight_2d.reshape(4096)
    
    N = 8192
    D = 4096
    
    x = add_62_recomputed.reshape(N, D)
    
    # Forward RMSNorm
    y = torch.empty((N, D), dtype=torch.bfloat16, device=device)
    rstd = torch.empty((N,), dtype=torch.float32, device=device)
    
    BLOCK_SIZE = 4096
    _rmsnorm_fwd_kernel[(N,)](
        x, weight, y, rstd,
        N, D, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Add mm_230 + mm_232 in bf16 first (to match reference precision)
    go = torch.empty((N, D), dtype=torch.bfloat16, device=device)
    total = N * D
    ADD_BLOCK = 1024
    grid_add = (triton.cdiv(total, ADD_BLOCK),)
    _add_bf16_kernel[grid_add](
        mm_230, mm_232, go,
        total,
        BLOCK_SIZE=ADD_BLOCK,
    )
    
    # Backward dx fused with add of getitem_420
    add_out = torch.empty((1, 8192, 4096), dtype=torch.bfloat16, device=device)
    g420 = getitem_420.reshape(N, D)
    add_out_2d = add_out.view(N, D)
    
    _rmsnorm_bwd_dx_kernel[(N,)](
        go, x, weight, rstd,
        g420, add_out_2d,
        N, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Backward dw (uses summed go)
    dw = torch.empty((D,), dtype=torch.float32, device=device)
    BLOCK_D = 128
    BLOCK_N = 64
    grid_dw = (triton.cdiv(D, BLOCK_D),)
    _rmsnorm_bwd_dw_kernel[grid_dw](
        go, x, rstd, dw,
        N, D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return (y, y, add_out, dw)