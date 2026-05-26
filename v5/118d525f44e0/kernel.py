import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    N: tl.constexpr, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    y = x * rstd * w
    tl.store(y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    x_ptr, w_ptr, rstd_ptr,
    mm0_ptr, mm1_ptr, mm2_ptr,
    add220_ptr,
    grad_in_ptr, grad_w_partial_ptr,
    M, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    # Load grads as bf16 then add with bf16 precision (matches PyTorch bf16 add semantics)
    g0 = tl.load(mm0_ptr + row * N + cols, mask=mask, other=0.0)
    g1 = tl.load(mm1_ptr + row * N + cols, mask=mask, other=0.0)
    g2 = tl.load(mm2_ptr + row * N + cols, mask=mask, other=0.0)
    # First add in fp32 then round to bf16 (simulating bf16 add result)
    s1 = (g0.to(tl.float32) + g1.to(tl.float32)).to(tl.bfloat16)
    s2 = (s1.to(tl.float32) + g2.to(tl.float32)).to(tl.bfloat16)
    grad_y = s2.to(tl.float32)

    # grad_w partial for this row = grad_y * (x * rstd) ; note: x*rstd is the normalized x
    x_hat = x * rstd
    gw_row = grad_y * x_hat
    tl.store(grad_w_partial_ptr + row * N + cols, gw_row, mask=mask)

    # grad_x = rstd * (grad_y * w - x_hat * mean(grad_y * w * x_hat))
    gw = grad_y * w
    c = tl.sum(gw * x_hat) / N
    grad_x = (gw - x_hat * c) * rstd

    # add_tensor_2 = add_220 + grad_x  -> bf16 add semantics
    a220 = tl.load(add220_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # grad_x from rms_norm_backward returns bf16, so cast first
    grad_x_bf = grad_x.to(tl.bfloat16).to(tl.float32)
    out = (a220 + grad_x_bf).to(tl.bfloat16)
    tl.store(grad_in_ptr + row * N + cols, out, mask=mask)


@triton.jit
def reduce_grad_w_kernel(
    grad_w_partial_ptr, grad_w_ptr,
    M, N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        mask = rows < M
        vals = tl.load(grad_w_partial_ptr + rows * N + col, mask=mask, other=0.0)
        acc += tl.sum(vals)
    tl.store(grad_w_ptr + col, acc)


@triton.jit
def embedding_backward_kernel(
    grad_out_ptr, indices_ptr, grad_weight_ptr,
    M, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    idx = tl.load(indices_ptr + row).to(tl.int64)
    g = tl.load(grad_out_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(grad_weight_ptr + idx * N + cols, g, mask=mask)


def kernel_function(getitem_791, embedding, mm_670, mm_672, mm_674, add_220, arg583_1):
    w_view = getitem_791.view(torch.bfloat16).contiguous().view(-1)
    assert w_view.numel() == 4096

    N = 4096
    M = 8192
    V = 128256

    emb_flat = embedding.contiguous().view(M, N)
    mm0 = mm_670.contiguous().view(M, N)
    mm1 = mm_672.contiguous().view(M, N)
    mm2 = mm_674.contiguous().view(M, N)
    add220_flat = add_220.contiguous().view(M, N)
    indices_flat = arg583_1.view(-1).contiguous()

    y = torch.empty((M, N), dtype=torch.bfloat16, device=embedding.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=embedding.device)
    grad_in = torch.empty((M, N), dtype=torch.bfloat16, device=embedding.device)
    grad_w_partial = torch.empty((M, N), dtype=torch.float32, device=embedding.device)
    grad_w = torch.empty((N,), dtype=torch.float32, device=embedding.device)
    grad_emb = torch.zeros((V, N), dtype=torch.float32, device=embedding.device)

    BLOCK_N = 4096
    eps = 1e-5

    rmsnorm_fwd_kernel[(M,)](
        emb_flat, w_view, y, rstd,
        N=N, eps=eps, BLOCK_N=BLOCK_N,
    )

    rmsnorm_bwd_kernel[(M,)](
        emb_flat, w_view, rstd,
        mm0, mm1, mm2,
        add220_flat,
        grad_in, grad_w_partial,
        M, N=N, BLOCK_N=BLOCK_N,
    )

    reduce_grad_w_kernel[(N,)](
        grad_w_partial, grad_w,
        M, N=N, BLOCK_M=256,
    )

    embedding_backward_kernel[(M,)](
        grad_in, indices_flat, grad_emb,
        M, N=N, BLOCK_N=BLOCK_N,
    )

    return (y, y, y, grad_w, grad_emb)