import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    mm3_ptr, emb_ptr, w_ptr,
    mm664_ptr, mm666_ptr, add218_ptr,
    rms_out_ptr, grad_input_ptr, grad_w_ptr,
    M, N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    row_off = row * N + offs

    mm3 = tl.load(mm3_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
    emb = tl.load(emb_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
    x = (mm3 + emb).to(tl.bfloat16).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xhat = x * rstd
    y = xhat * w
    tl.store(rms_out_ptr + row_off, y.to(tl.bfloat16), mask=mask)

    mm664 = tl.load(mm664_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
    mm666 = tl.load(mm666_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
    grad_y = (mm664 + mm666).to(tl.bfloat16).to(tl.float32)

    add218 = tl.load(add218_ptr + row_off, mask=mask, other=0.0).to(tl.float32)

    gw_contrib = grad_y * xhat
    gyw = grad_y * w
    s = tl.sum(gyw * xhat, axis=0) / N
    grad_x = rstd * (gyw - xhat * s)
    out_val = add218 + grad_x

    tl.store(grad_input_ptr + row_off, out_val.to(tl.bfloat16), mask=mask)
    tl.atomic_add(grad_w_ptr + offs, gw_contrib, mask=mask)


def kernel_function(mm_3, embedding, getitem_787, mm_664, mm_666, add_218):
    M = 8192
    N = 4096
    eps = 1e-5

    weight = getitem_787.contiguous().view(N)
    device = mm_3.device

    rms_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    grad_input = torch.empty((1, M, N), dtype=torch.bfloat16, device=device)
    grad_w_out = torch.zeros((N,), dtype=torch.float32, device=device)

    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _fused_kernel[grid](
        mm_3, embedding, weight,
        mm_664, mm_666, add_218,
        rms_out, grad_input, grad_w_out,
        M, N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=16,
        num_stages=2,
    )

    return (rms_out, rms_out, grad_input, grad_w_out)