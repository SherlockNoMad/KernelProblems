import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, M, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    y_row = y_ptr + row * N

    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(rstd_ptr + row, rstd)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_row + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def fused_add3_rms_bwd_dx_kernel(
    a_ptr, b_ptr, c_ptr,
    x_ptr, w_ptr, rstd_ptr,
    add220_ptr,
    out_add_ptr,
    partial_dw_ptr,
    M, N, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    sum_dyw_x = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(a_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        # Match reference: add bf16 step by step with bf16 rounding
        ab = (a + b)
        ab_bf = ab.to(tl.bfloat16).to(tl.float32)
        abc = ab_bf + c
        dy = abc.to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_dyw_x += tl.sum(dy * w * x)

    coef = rstd * rstd * rstd / N

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(a_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        ab = (a + b)
        ab_bf = ab.to(tl.bfloat16).to(tl.float32)
        abc = ab_bf + c
        dy = abc.to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        dx = rstd * dy * w - x * coef * sum_dyw_x
        # Round dx to bf16 to match reference output dtype
        dx_bf = dx.to(tl.bfloat16).to(tl.float32)
        add220 = tl.load(add220_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        out = add220 + dx_bf
        tl.store(out_add_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)

        # Partial dw contribution: dy * x * rstd, rounded to bf16 (the weight grad is bf16 then cast to fp32)
        pdw = dy * x * rstd
        tl.store(partial_dw_ptr + row * N + cols, pdw, mask=mask)


@triton.jit
def reduce_dw_kernel(partial_dw_ptr, dw_ptr, M, N, BLOCK_M: tl.constexpr):
    col = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        rows = m_start + tl.arange(0, BLOCK_M)
        mask = rows < M
        vals = tl.load(partial_dw_ptr + rows * N + col, mask=mask, other=0.0)
        acc += tl.sum(vals)
    # Cast accumulated sum to bf16 then back to fp32 to match reference (weight grad is bf16)
    acc_bf = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(dw_ptr + col, acc_bf)


@triton.jit
def embedding_backward_kernel(
    grad_out_ptr, indices_ptr, grad_weight_ptr,
    num_tokens, embed_dim, BLOCK: tl.constexpr
):
    token_id = tl.program_id(0)
    idx = tl.load(indices_ptr + token_id)
    for off in range(0, embed_dim, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < embed_dim
        g = tl.load(grad_out_ptr + token_id * embed_dim + cols, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(grad_weight_ptr + idx * embed_dim + cols, g, mask=mask)


def kernel_function(embedding, _unsafe_view_432, mm_670, mm_672, mm_674, add_220, arg583_1):
    assert embedding.is_cuda
    B, S, D = embedding.shape
    M = B * S
    N = D

    y = torch.empty_like(embedding)
    rstd = torch.empty((M,), dtype=torch.float32, device=embedding.device)

    BLOCK = 1024
    rms_norm_fwd_kernel[(M,)](
        embedding, _unsafe_view_432, y, rstd, M, N, 1e-5, BLOCK=BLOCK
    )

    view_default = y.view(M, N)
    view_default_1 = y.view(M, N)
    view_default_2 = y.view(M, N)

    add_tensor_2 = torch.empty_like(add_220)
    partial_dw = torch.empty((M, N), dtype=torch.float32, device=embedding.device)

    fused_add3_rms_bwd_dx_kernel[(M,)](
        mm_670, mm_672, mm_674,
        embedding, _unsafe_view_432, rstd,
        add_220,
        add_tensor_2,
        partial_dw,
        M, N, BLOCK=BLOCK
    )

    dw_fp32 = torch.empty((N,), dtype=torch.float32, device=embedding.device)
    reduce_dw_kernel[(N,)](partial_dw, dw_fp32, M, N, BLOCK_M=128)

    to_copy_default = dw_fp32

    num_embeddings = 128256
    grad_weight = torch.zeros((num_embeddings, N), dtype=torch.float32, device=embedding.device)

    num_tokens = arg583_1.numel()
    indices_flat = arg583_1.contiguous().view(-1)
    grad_out_flat = add_tensor_2.view(num_tokens, N)

    embedding_backward_kernel[(num_tokens,)](
        grad_out_flat, indices_flat, grad_weight,
        num_tokens, N, BLOCK=1024
    )

    to_copy_default_1 = grad_weight

    return (view_default, view_default_1, view_default_2, to_copy_default, to_copy_default_1)