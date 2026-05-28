import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_bwd_kernel(
    mm0_ptr, mm1_ptr, mm2_ptr,  # bf16 [8192, 4096]
    embedding_ptr,  # bf16 [1,8192,4096]
    weight_ptr,  # bf16 [4096]
    rstd_ptr,  # fp32 [1,8192,1]
    add220_ptr,  # bf16 [1,8192,4096]
    intermediate_ptr,  # bf16 [1,8192,4096] output: grad input + add220
    grad_weight_ptr,  # fp32 [4096] - accumulated via atomic add
    M,  # 8192
    N: tl.constexpr,  # 4096
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    row_offset = row * N

    # Load three mm contributions and emulate bf16 add chain:
    # add_tensor   = (mm0 + mm1)    [bf16]
    # add_tensor_1 = (add_tensor + mm2) [bf16]
    mm0 = tl.load(mm0_ptr + row_offset + cols, mask=mask, other=0.0)
    mm1 = tl.load(mm1_ptr + row_offset + cols, mask=mask, other=0.0)
    mm2 = tl.load(mm2_ptr + row_offset + cols, mask=mask, other=0.0)

    mm0_f = mm0.to(tl.float32)
    mm1_f = mm1.to(tl.float32)
    mm2_f = mm2.to(tl.float32)

    add01 = (mm0_f + mm1_f).to(tl.bfloat16).to(tl.float32)
    dy = (add01 + mm2_f).to(tl.bfloat16).to(tl.float32)

    x = tl.load(embedding_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dy_w = dy * w

    c = tl.sum(dy_w * x_hat, axis=0) / N

    grad_input = rstd * (dy_w - x_hat * c)
    # grad_input is bf16 in reference (output of _fused_rms_norm_backward)
    grad_input_bf16 = grad_input.to(tl.bfloat16).to(tl.float32)

    add220 = tl.load(add220_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    out = add220 + grad_input_bf16

    tl.store(intermediate_ptr + row_offset + cols, out.to(tl.bfloat16), mask=mask)

    # grad_weight accumulation: sum over rows of dy * x_hat
    # Reference accumulates in fp32 then casts to bf16 at the end.
    gw_contrib = dy * x_hat
    tl.atomic_add(grad_weight_ptr + cols, gw_contrib, mask=mask)


@triton.jit
def _embedding_dense_backward_kernel(
    grad_output_ptr,  # bf16 [1,8192,4096]
    indices_ptr,  # int64 [1,8192]
    grad_weight_ptr,  # fp32 [num_embeddings, 4096]
    num_tokens,  # 8192
    num_embeddings,  # 128256
    N: tl.constexpr,  # 4096
    BLOCK_N: tl.constexpr,
):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    idx = tl.load(indices_ptr + tok)
    valid = (idx >= 0) & (idx < num_embeddings)

    go = tl.load(grad_output_ptr + tok * N + cols, mask=mask, other=0.0).to(tl.float32)

    if valid:
        tl.atomic_add(grad_weight_ptr + idx * N + cols, go, mask=mask)


@triton.jit
def _cast_bf16_kernel(
    in_ptr,      # fp32 [N]
    out_ptr,     # fp32 [N] but rounded to bf16
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(in_ptr + offs, mask=mask, other=0.0)
    v_bf16 = v.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs, v_bf16, mask=mask)


def kernel_function(
    _fused_rms_norm_recomputed,
    mm_670,
    mm_672,
    mm_674,
    embedding,
    _unsafe_view_432,
    add_220,
    arg583_1,
):
    rstd = _fused_rms_norm_recomputed[1]  # fp32 [1,8192,1]

    assert embedding.is_cuda
    M = 8192
    N = 4096
    num_embeddings = 128256

    grad_input_plus_add = torch.empty((1, M, N), dtype=torch.bfloat16, device=embedding.device)
    grad_weight_rms_fp32 = torch.zeros((N,), dtype=torch.float32, device=embedding.device)

    BLOCK_N = 4096
    grid_rms = (M,)
    _rmsnorm_bwd_kernel[grid_rms](
        mm_670, mm_672, mm_674,
        embedding,
        _unsafe_view_432,
        rstd,
        add_220,
        grad_input_plus_add,
        grad_weight_rms_fp32,
        M,
        N=N,
        BLOCK_N=BLOCK_N,
    )

    # Simulate the bf16 cast that occurs at the end of _fused_rms_norm_backward
    # (its grad_weight output is bf16), followed by _to_copy to fp32.
    grad_weight_out = torch.empty((N,), dtype=torch.float32, device=embedding.device)
    BLOCK_CAST = 256
    grid_cast = (triton.cdiv(N, BLOCK_CAST),)
    _cast_bf16_kernel[grid_cast](
        grad_weight_rms_fp32,
        grad_weight_out,
        N,
        BLOCK=BLOCK_CAST,
    )

    # Embedding dense backward
    grad_embedding_weight = torch.zeros(
        (num_embeddings, N), dtype=torch.float32, device=embedding.device
    )
    indices = arg583_1.contiguous()
    grid_emb = (M,)
    _embedding_dense_backward_kernel[grid_emb](
        grad_input_plus_add,
        indices,
        grad_embedding_weight,
        M,
        num_embeddings,
        N=N,
        BLOCK_N=BLOCK_N,
    )

    out0 = grad_weight_out
    out1 = grad_embedding_weight

    return (out0, out1)