import torch
import triton
import triton.language as tl


@triton.jit
def _fused_forward_kernel(
    mm_3_ptr, embedding_ptr, weight_ptr,
    out1_ptr, out2_ptr,
    add_tensor_ptr, rstd_ptr,
    hidden_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    for row_idx in range(ROWS_PER_BLOCK):
        row = row_start + row_idx
        row_offset = row * hidden_dim

        mm_3_vals = tl.load(mm_3_ptr + row_offset + offsets, mask=mask, other=0.0)
        embedding_vals = tl.load(embedding_ptr + row_offset + offsets, mask=mask, other=0.0)

        add_tensor_vals = mm_3_vals + embedding_vals

        tl.store(add_tensor_ptr + row_offset + offsets, add_tensor_vals, mask=mask)

        add_tensor_f32 = add_tensor_vals.to(tl.float32)
        sum_sq = tl.sum(add_tensor_f32 * add_tensor_f32)
        mean_sq = sum_sq / hidden_dim
        rstd_val = 1.0 / tl.sqrt(mean_sq + eps)
        tl.store(rstd_ptr + row, rstd_val)

        weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        normalized = add_tensor_f32 * rstd_val * weight_vals.to(tl.float32)
        normalized_bf16 = normalized.to(tl.bfloat16)

        tl.store(out1_ptr + row_offset + offsets, normalized_bf16, mask=mask)
        tl.store(out2_ptr + row_offset + offsets, normalized_bf16, mask=mask)


@triton.jit
def _fused_backward_kernel(
    mm_664_ptr, mm_666_ptr, add_tensor_ptr, weight_ptr, add_218_ptr,
    rstd_ptr,
    out3_ptr,
    wgrad_partial_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    num_rows: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    wgrad_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    weight_f32 = weight_vals.to(tl.float32)

    for row_idx in range(ROWS_PER_BLOCK):
        row = row_start + row_idx
        if row < num_rows:
            row_offset = row * hidden_dim

            rstd_val = tl.load(rstd_ptr + row)

            mm_664_vals = tl.load(mm_664_ptr + row_offset + offsets, mask=mask, other=0.0)
            mm_666_vals = tl.load(mm_666_ptr + row_offset + offsets, mask=mask, other=0.0)
            add_tensor_vals = tl.load(add_tensor_ptr + row_offset + offsets, mask=mask, other=0.0)

            grad_output_f32 = (mm_664_vals + mm_666_vals).to(tl.float32)
            add_tensor_f32 = add_tensor_vals.to(tl.float32)

            sum_grad_weight = tl.sum(tl.where(mask, grad_output_f32 * weight_f32 * add_tensor_f32, 0.0))
            correction_factor = sum_grad_weight * rstd_val * rstd_val * rstd_val / hidden_dim

            grad_input = rstd_val * weight_f32 * grad_output_f32 - correction_factor * add_tensor_f32

            add_218_vals = tl.load(add_218_ptr + row_offset + offsets, mask=mask, other=0.0)
            final_output = add_218_vals + grad_input.to(tl.bfloat16)
            tl.store(out3_ptr + row_offset + offsets, final_output, mask=mask)

            wgrad_acc += grad_output_f32 * add_tensor_f32 * rstd_val

    tl.atomic_add(wgrad_partial_ptr + offsets, wgrad_acc, mask=mask)


def kernel_function(mm_3, embedding, getitem_787, mm_664, mm_666, add_218):
    device = mm_3.device
    seq_len = 8192
    hidden_dim = 4096
    batch_size = 1

    mm_3 = mm_3.contiguous()
    embedding = embedding.contiguous().view(seq_len, hidden_dim)
    mm_664 = mm_664.contiguous()
    mm_666 = mm_666.contiguous()
    add_218 = add_218.contiguous().view(seq_len, hidden_dim)
    weight = getitem_787.contiguous().view(hidden_dim)

    out1 = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    out2 = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    out3 = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    out4 = torch.zeros([hidden_dim], dtype=torch.float32, device=device)
    add_tensor = torch.empty([seq_len, hidden_dim], dtype=torch.bfloat16, device=device)
    rstd = torch.empty([seq_len], dtype=torch.float32, device=device)

    BLOCK_SIZE = 4096
    ROWS_PER_BLOCK_FWD = 4
    ROWS_PER_BLOCK_BWD = 8

    grid_fwd = (seq_len // ROWS_PER_BLOCK_FWD,)
    grid_bwd = ((seq_len + ROWS_PER_BLOCK_BWD - 1) // ROWS_PER_BLOCK_BWD,)

    _fused_forward_kernel[grid_fwd](
        mm_3, embedding, weight,
        out1, out2,
        add_tensor, rstd,
        hidden_dim, 1e-05,
        BLOCK_SIZE, ROWS_PER_BLOCK_FWD,
        num_warps=8,
    )

    _fused_backward_kernel[grid_bwd](
        mm_664, mm_666, add_tensor, weight, add_218,
        rstd,
        out3, out4,
        hidden_dim, BLOCK_SIZE, ROWS_PER_BLOCK_BWD, seq_len,
        num_warps=8,
    )

    return (out1, out2, out3.view(batch_size, seq_len, hidden_dim), out4)