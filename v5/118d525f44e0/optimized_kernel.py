import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fused_kernel(
    x_ptr, w_ptr,
    mm0_ptr, mm1_ptr, mm2_ptr,
    add220_ptr,
    y_ptr, grad_in_ptr, grad_w_ptr,
    N: tl.constexpr, eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    M: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    grad_w_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    row_start = pid * ROWS_PER_PROG
    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            off = row * N + cols
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x) / N
            rstd = 1.0 / tl.sqrt(var + eps)
            x_hat = x * rstd
            y = x_hat * w
            tl.store(y_ptr + off, y.to(tl.bfloat16), mask=mask)

            g0 = tl.load(mm0_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g1 = tl.load(mm1_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g2 = tl.load(mm2_ptr + off, mask=mask, other=0.0).to(tl.float32)
            s1 = (g0 + g1).to(tl.bfloat16).to(tl.float32)
            grad_y = (s1 + g2).to(tl.bfloat16).to(tl.float32)

            grad_w_acc += grad_y * x_hat

            gw = grad_y * w
            c = tl.sum(gw * x_hat) / N
            grad_x = (gw - x_hat * c) * rstd

            a220 = tl.load(add220_ptr + off, mask=mask, other=0.0).to(tl.float32)
            grad_x_bf = grad_x.to(tl.bfloat16).to(tl.float32)
            out = (a220 + grad_x_bf).to(tl.bfloat16)
            tl.store(grad_in_ptr + off, out, mask=mask)

    tl.atomic_add(grad_w_ptr + cols, grad_w_acc, mask=mask)


@triton.jit
def embedding_backward_kernel(
    grad_out_ptr, indices_ptr, grad_weight_ptr,
    N: tl.constexpr, BLOCK_N: tl.constexpr,
    M: tl.constexpr,
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
    grad_in = torch.empty((M, N), dtype=torch.bfloat16, device=embedding.device)
    grad_w = torch.zeros((N,), dtype=torch.float32, device=embedding.device)
    grad_emb = torch.zeros((V, N), dtype=torch.float32, device=embedding.device)

    BLOCK_N = 4096
    eps = 1e-5
    ROWS_PER_PROG = 2
    grid_m = (M + ROWS_PER_PROG - 1) // ROWS_PER_PROG

    rmsnorm_fused_kernel[(grid_m,)](
        emb_flat, w_view,
        mm0, mm1, mm2,
        add220_flat,
        y, grad_in, grad_w,
        N=N, eps=eps, BLOCK_N=BLOCK_N,
        ROWS_PER_PROG=ROWS_PER_PROG, M=M,
        num_warps=8,
        num_stages=3,
    )

    embedding_backward_kernel[(M,)](
        grad_in, indices_flat, grad_emb,
        N=N, BLOCK_N=BLOCK_N, M=M,
        num_warps=8,
    )

    return (y, y, y, grad_w, grad_emb)