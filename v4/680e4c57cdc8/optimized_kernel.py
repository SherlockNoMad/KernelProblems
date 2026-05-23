import torch
import triton
import triton.language as tl


@triton.jit
def fused_rms_norm_fwd_bwd_kernel(
    embedding_ptr,
    weight_ptr,
    mm_670_ptr,
    mm_672_ptr,
    mm_674_ptr,
    add_220_ptr,
    indices_ptr,
    normalized_out_ptr,
    grad_weight_ptr,
    embedding_grad_ptr,
    hidden_dim: tl.constexpr,
    vocab_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load embedding and weight once
    input_data = tl.load(embedding_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    weight_data = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS norm
    mean_sq = tl.sum(input_data * input_data, axis=0) / hidden_dim
    inv_rss = 1.0 / tl.sqrt(mean_sq + eps)

    x_normalized = input_data * inv_rss
    normalized = x_normalized * weight_data

    # Store normalized output
    tl.store(normalized_out_ptr + base_offset + offsets, normalized.to(tl.bfloat16), mask=mask)

    # Load gradients and compute sum
    g0 = tl.load(mm_670_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    g1 = tl.load(mm_672_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(mm_674_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_output = g0 + g1 + g2

    # Backward: grad_weight
    grad_weight_local = grad_output * x_normalized

    # Backward: grad_input
    grad_norm_prod = grad_output * weight_data * x_normalized
    mean_grad_norm = tl.sum(grad_norm_prod, axis=0) / hidden_dim
    grad_input = inv_rss * (grad_output * weight_data - mean_grad_norm * x_normalized)

    # Add residual
    add_220_data = tl.load(add_220_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    final_grad = add_220_data + grad_input

    # Atomic updates
    tl.atomic_add(grad_weight_ptr + offsets, grad_weight_local, mask=mask)

    token_idx = tl.load(indices_ptr + pid)
    embedding_offset = token_idx * hidden_dim + offsets
    tl.atomic_add(embedding_grad_ptr + embedding_offset, final_grad, mask=mask)


def kernel_function(getitem_791, embedding, mm_670, mm_672, mm_674, add_220, arg583_1):
    device = embedding.device
    batch_size, seq_len, hidden_dim = embedding.shape
    total_seqs = batch_size * seq_len
    vocab_size = 128256
    eps = 1e-5

    getitem_flat = getitem_791.contiguous().view(-1)
    weight_data = getitem_flat[:hidden_dim].contiguous()

    normalized_data = torch.empty((total_seqs, hidden_dim), dtype=torch.bfloat16, device=device)
    grad_weight = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
    embedding_grad = torch.zeros((vocab_size, hidden_dim), dtype=torch.float32, device=device)

    indices_flat = arg583_1.view(-1)

    BLOCK_SIZE = 4096

    fused_rms_norm_fwd_bwd_kernel[(total_seqs,)](
        embedding,
        weight_data,
        mm_670,
        mm_672,
        mm_674,
        add_220,
        indices_flat,
        normalized_data,
        grad_weight,
        embedding_grad,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        num_stages=1,
    )

    reshape_out = normalized_data.view(total_seqs, hidden_dim)

    return (reshape_out, reshape_out, reshape_out, grad_weight, embedding_grad)