import torch
import triton
import triton.language as tl


@triton.jit
def gather_weight_kernel(src_ptr, dst_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def rms_norm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    N: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * N
    y_ptr += row * N

    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x * x
    mean = tl.sum(_sum) / N
    rstd = 1.0 / tl.sqrt(mean + eps)
    tl.store(rstd_ptr + row, rstd)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_ptr + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rms_norm_bwd_dx_kernel(
    a_ptr, b_ptr, c_ptr,
    x_ptr, w_ptr, rstd_ptr,
    add220_ptr,
    dx_out_ptr,
    dw_partial_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    a_ptr += row * N
    b_ptr += row * N
    c_ptr += row * N
    x_ptr += row * N
    add220_ptr += row * N
    dx_out_ptr += row * N
    dw_partial_ptr += row * N

    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    # Compute dy in bf16-accumulated way to match PyTorch:
    # view_default_3 + view_default_4 (bf16) -> add_tensor (bf16)
    # add_tensor + view_default_5 (bf16) -> add_tensor_1 (bf16)
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(a_ptr + cols, mask=mask, other=0.0)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0)
        c = tl.load(c_ptr + cols, mask=mask, other=0.0)
        # Match PyTorch sequence: bf16 add then bf16 add
        ab = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        dy_bf16 = (ab.to(tl.float32) + c.to(tl.float32)).to(tl.bfloat16)
        dy = dy_bf16.to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += dy * w * x
    c1 = tl.sum(_sum) / N

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(a_ptr + cols, mask=mask, other=0.0)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0)
        c = tl.load(c_ptr + cols, mask=mask, other=0.0)
        ab = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        dy_bf16 = (ab.to(tl.float32) + c.to(tl.float32)).to(tl.bfloat16)
        dy = dy_bf16.to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        add220 = tl.load(add220_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        dx = rstd * (dy * w - x_hat * c1 * rstd)
        # match PyTorch: dx is bf16, then add_220 + dx in bf16
        dx_bf16 = dx.to(tl.bfloat16)
        out = (add220 + dx_bf16.to(tl.float32)).to(tl.bfloat16)
        tl.store(dx_out_ptr + cols, out, mask=mask)
        dw_row = dy * x_hat
        tl.store(dw_partial_ptr + cols, dw_row, mask=mask)


@triton.jit
def dw_reduce_kernel(dw_partial_ptr, dw_ptr, M, N: tl.constexpr, BLOCK_M: tl.constexpr):
    col = tl.program_id(0)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for off in range(0, M, BLOCK_M):
        rows = off + tl.arange(0, BLOCK_M)
        mask = rows < M
        vals = tl.load(dw_partial_ptr + rows * N + col, mask=mask, other=0.0)
        acc += vals
    s = tl.sum(acc)
    tl.store(dw_ptr + col, s)


@triton.jit
def embedding_backward_kernel(
    grad_out_ptr, indices_ptr, grad_weight_ptr,
    num_indices, embed_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_idx = tl.program_id(0)
    pid_d = tl.program_id(1)
    idx = tl.load(indices_ptr + pid_idx)
    cols = pid_d * BLOCK + tl.arange(0, BLOCK)
    mask = cols < embed_dim
    grad = tl.load(grad_out_ptr + pid_idx * embed_dim + cols, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(grad_weight_ptr + idx * embed_dim + cols, grad, mask=mask)


def kernel_function(getitem_791, embedding, mm_670, mm_672, mm_674, add_220, arg583_1):
    device = embedding.device
    M = 8192
    N = 4096
    eps = 1e-5

    weight = torch.empty(4096, dtype=torch.bfloat16, device=device)
    BLOCK = 256
    gather_weight_kernel[(triton.cdiv(4096, BLOCK),)](getitem_791, weight, 4096, BLOCK=BLOCK)

    x = embedding.view(M, N)
    y = torch.empty_like(x)
    rstd = torch.empty(M, dtype=torch.float32, device=device)
    rms_norm_fwd_kernel[(M,)](x, weight, y, rstd, N, eps, BLOCK=1024)

    view_default = y
    view_default_1 = y
    view_default_2 = y

    grad_x_full = torch.empty_like(embedding).view(M, N)
    dw_partial = torch.empty((M, N), dtype=torch.float32, device=device)
    a_flat = mm_670.view(M, N)
    b_flat = mm_672.view(M, N)
    c_flat = mm_674.view(M, N)
    add220_flat = add_220.view(M, N)
    rms_norm_bwd_dx_kernel[(M,)](
        a_flat, b_flat, c_flat,
        x, weight, rstd,
        add220_flat,
        grad_x_full,
        dw_partial,
        N, BLOCK=1024,
    )

    dw_fp32 = torch.empty(N, dtype=torch.float32, device=device)
    dw_reduce_kernel[(N,)](dw_partial, dw_fp32, M, N, BLOCK_M=128)

    grad_weight = torch.zeros((128256, N), dtype=torch.float32, device=device)
    indices_flat = arg583_1.view(-1).contiguous()
    num_indices = indices_flat.numel()
    grad_out_flat = grad_x_full
    BLOCK_D = 256
    embedding_backward_kernel[(num_indices, triton.cdiv(N, BLOCK_D))](
        grad_out_flat, indices_flat, grad_weight,
        num_indices, N, BLOCK=BLOCK_D,
    )

    return (view_default, view_default_1, view_default_2, dw_fp32, grad_weight)