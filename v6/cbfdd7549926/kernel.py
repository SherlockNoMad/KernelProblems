import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    mm3_ptr, emb_ptr, w_ptr,
    x_added_ptr, out_ptr, rstd_ptr,
    M, N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    
    mm3 = tl.load(mm3_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    emb = tl.load(emb_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    # Round x through bf16 to match reference (add_tensor is bf16)
    x_bf16 = (mm3 + emb).to(tl.bfloat16)
    x = x_bf16.to(tl.float32)
    
    # Store x_added in bf16
    tl.store(x_added_ptr + row * N + offs, x_bf16, mask=mask)
    
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)
    
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_kernel(
    mm664_ptr, mm666_ptr, x_ptr, w_ptr, rstd_ptr, add218_ptr,
    out_grad_input_ptr, grad_w_ptr,
    M, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    
    # grad_y = mm_664 + mm_666, but ref does bf16 add then bf16 -> internal fp32
    mm664 = tl.load(mm664_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mm666 = tl.load(mm666_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    grad_y_bf16 = (mm664 + mm666).to(tl.bfloat16)
    grad_y = grad_y_bf16.to(tl.float32)
    
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)
    add218 = tl.load(add218_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    
    # grad_w contribution = grad_y * x * rstd
    gw_contrib = grad_y * x * rstd
    
    # s = mean(grad_y * w * x)
    gyw_x = grad_y * w * x
    s = tl.sum(gyw_x, axis=0) / N
    
    grad_x = rstd * (grad_y * w - x * rstd * rstd * s)
    
    out_val = add218 + grad_x
    tl.store(out_grad_input_ptr + row * N + offs, out_val.to(tl.bfloat16), mask=mask)
    
    # Atomic add to grad_w (fp32)
    tl.atomic_add(grad_w_ptr + offs, gw_contrib, mask=mask)


@triton.jit
def _cast_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # Round-trip through bf16 to mimic getitem_790 dtype
    val_bf16 = val.to(tl.bfloat16)
    tl.store(out_ptr + offs, val_bf16.to(tl.float32), mask=mask)


def kernel_function(mm_3, embedding, getitem_787, mm_664, mm_666, add_218):
    """
    Fused RMS norm forward + backward.
    """
    assert mm_3.is_cuda
    M = 8192
    N = 4096
    eps = 1e-5
    
    # Weight: getitem_787 cloned contiguous, viewed as [4096]
    weight = getitem_787.contiguous().view(N)
    
    device = mm_3.device
    x_added = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    rms_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    rstd = torch.empty((M,), dtype=torch.float32, device=device)
    
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _fwd_kernel[grid](
        mm_3, embedding, weight,
        x_added, rms_out, rstd,
        M, N, eps,
        BLOCK_N=BLOCK_N,
    )
    
    grad_input = torch.empty((1, M, N), dtype=torch.bfloat16, device=device)
    grad_w_fp32 = torch.zeros((N,), dtype=torch.float32, device=device)
    
    _bwd_kernel[grid](
        mm_664, mm_666, x_added, weight, rstd, add_218,
        grad_input, grad_w_fp32,
        M, N,
        BLOCK_N=BLOCK_N,
    )
    
    # Round-trip grad_w through bf16 to match reference (getitem_790 is bf16)
    grad_w_out = torch.empty((N,), dtype=torch.float32, device=device)
    BLOCK_CAST = 1024
    _cast_kernel[(triton.cdiv(N, BLOCK_CAST),)](grad_w_fp32, grad_w_out, N, BLOCK=BLOCK_CAST)
    
    return (rms_out, rms_out, grad_input, grad_w_out)