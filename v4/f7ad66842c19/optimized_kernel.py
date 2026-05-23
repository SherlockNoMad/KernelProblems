import torch
import triton
import triton.language as tl


@triton.jit
def fused_rms_norm_bwd_kernel(
    mm_223_ptr, add_62_ptr, mm_226_ptr, weight_ptr,
    out1_ptr, out2_ptr,
    seq_len,
    eps: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    block_idx = tl.program_id(0)
    total_rows = seq_len

    row_start = block_idx * ROWS_PER_BLOCK
    offsets = tl.arange(0, HIDDEN_DIM)

    # Accumulate weight gradient contributions across multiple rows
    grad_w_accum = tl.zeros([HIDDEN_DIM], dtype=tl.float32)
    w = tl.load(weight_ptr + offsets).to(tl.float32)

    for row_offset in range(ROWS_PER_BLOCK):
        row_idx = row_start + row_offset
        if row_idx < total_rows:
            seq_offset = row_idx * HIDDEN_DIM

            # Load all data for this row
            mm_223_val = tl.load(mm_223_ptr + seq_offset + offsets)
            add_62_val = tl.load(add_62_ptr + seq_offset + offsets)
            x = (add_62_val + mm_223_val).to(tl.float32)

            # Compute RMS norm
            mean_sq = tl.sum(x * x) * (1.0 / HIDDEN_DIM)
            inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
            normalized = x * inv_rms

            # Load grad output
            g = tl.load(mm_226_ptr + seq_offset + offsets).to(tl.float32)

            # Compute grad_weight_sum
            gw = g * w
            grad_weight_sum = tl.sum(gw * normalized)
            coeff = grad_weight_sum * (1.0 / HIDDEN_DIM)

            # Compute and store grad_input
            grad_input = (gw - normalized * coeff) * inv_rms
            tl.store(out1_ptr + seq_offset + offsets, grad_input.to(tl.bfloat16))

            # Accumulate weight gradient
            grad_w_accum += g * normalized

    # Single write of accumulated weight gradients (instead of per-row atomics)
    tl.atomic_add(out2_ptr + offsets, grad_w_accum)


def kernel_function(mm_223, add_62_recomputed, mm_226, wait_tensor_970):
    device = mm_223.device
    batch_size = add_62_recomputed.shape[0]
    seq_len = add_62_recomputed.shape[1]
    hidden_dim = add_62_recomputed.shape[2]

    weight_final = wait_tensor_970.view(-1)

    out1 = torch.empty(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device=device)
    out2 = torch.zeros(hidden_dim, dtype=torch.float32, device=device)

    total_rows = batch_size * seq_len

    mm_223_flat = mm_223.view(-1)
    add_62_flat = add_62_recomputed.view(-1)
    mm_226_flat = mm_226.view(-1)
    out1_flat = out1.view(-1)

    ROWS_PER_BLOCK = 8
    num_blocks = (total_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    grid = (num_blocks,)
    fused_rms_norm_bwd_kernel[grid](
        mm_223_flat, add_62_flat, mm_226_flat, weight_final,
        out1_flat, out2,
        total_rows,
        eps=1e-5,
        HIDDEN_DIM=hidden_dim,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        num_warps=16,
        num_stages=1,
    )

    return (out1, out2)