import torch
import triton
import triton.language as tl


@triton.jit
def fused_rms_norm_fwd_write3_kernel(
    add_1_ptr, weight_ptr,
    out1_ptr, out2_ptr, out3_ptr,
    inv_rms_ptr,
    hidden_size,
    stride_input,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_input

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_size

    # Load input row
    x = tl.load(add_1_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS norm
    sum_squares = tl.sum(x * x, axis=0)
    mean_square = sum_squares / hidden_size
    rms = tl.sqrt(mean_square + 1e-5)
    inv_rms = 1.0 / rms

    # Store inv_rms as scalar
    tl.store(inv_rms_ptr + row_idx, inv_rms)

    # Load weight and normalize
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    normalized = x * inv_rms
    output = (normalized * weight).to(tl.bfloat16)

    # Write to 3 outputs (fused)
    out_row_start = row_idx * hidden_size
    tl.store(out1_ptr + out_row_start + col_offsets, output, mask=mask)
    tl.store(out2_ptr + out_row_start + col_offsets, output, mask=mask)
    tl.store(out3_ptr + out_row_start + col_offsets, output, mask=mask)


@triton.jit
def fused_rms_norm_backward_add_kernel(
    grad_output_a_ptr, grad_output_b_ptr, grad_output_c_ptr,
    input_ptr, inv_rms_ptr, weight_ptr,
    add_215_ptr,
    grad_input_add_ptr, grad_weight_ptr,
    hidden_size,
    stride_input,
    stride_grad_a, stride_grad_b, stride_grad_c,
    stride_add215,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_size

    inv_rms = tl.load(inv_rms_ptr + row_idx).to(tl.float32)

    # Load all grad outputs and sum them (fused)
    ga = tl.load(grad_output_a_ptr + row_idx * stride_grad_a + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gb = tl.load(grad_output_b_ptr + row_idx * stride_grad_b + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gc = tl.load(grad_output_c_ptr + row_idx * stride_grad_c + col_offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = ga + gb + gc

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    x = tl.load(input_ptr + row_idx * stride_input + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute grad through RMS norm
    grad_x_norm = grad_out * weight
    sum_grad_x_norm_x = tl.sum(grad_x_norm * x, axis=0)

    coeff = inv_rms * inv_rms * inv_rms * sum_grad_x_norm_x / hidden_size
    grad_input = inv_rms * grad_x_norm - coeff * x

    # Accumulate grad_weight
    x_norm = x * inv_rms
    grad_weight_local = (grad_out * x_norm).to(tl.float32)
    tl.atomic_add(grad_weight_ptr + col_offsets, grad_weight_local, mask=mask)

    # Fused add with add_215
    add_215_val = tl.load(add_215_ptr + row_idx * stride_add215 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    final_result = add_215_val + grad_input
    tl.store(grad_input_add_ptr + row_idx * stride_out + col_offsets, final_result.to(tl.bfloat16), mask=mask)


def kernel_function(getitem_818, add_1, mm_656, mm_658, mm_660, add_215):
    device = add_1.device

    # Determine shapes
    if add_1.dim() == 3:
        batch_size, seq_len, hidden_size = add_1.shape
        total_rows = batch_size * seq_len
    elif add_1.dim() == 2:
        total_rows, hidden_size = add_1.shape
        seq_len = total_rows
    else:
        raise ValueError("Unexpected add_1 dimensions")

    # Weight is getitem_818 (1D)
    weight = getitem_818

    # Choose BLOCK_SIZE as next power of 2 >= hidden_size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)

    grid = (total_rows,)

    out1 = torch.empty((total_rows, hidden_size), dtype=torch.bfloat16, device=device)
    out2 = torch.empty((total_rows, hidden_size), dtype=torch.bfloat16, device=device)
    out3 = torch.empty((total_rows, hidden_size), dtype=torch.bfloat16, device=device)
    inv_rms = torch.empty((total_rows,), dtype=torch.float32, device=device)

    add_1_flat = add_1.reshape(total_rows, hidden_size)
    stride_input = add_1_flat.stride(0)

    fused_rms_norm_fwd_write3_kernel[grid](
        add_1_flat, weight,
        out1, out2, out3,
        inv_rms,
        hidden_size,
        stride_input,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    final_result = torch.empty_like(add_215).reshape(total_rows, hidden_size)
    grad_weight = torch.zeros((hidden_size,), dtype=torch.float32, device=device)

    mm_656_flat = mm_656.reshape(total_rows, hidden_size)
    mm_658_flat = mm_658.reshape(total_rows, hidden_size)
    mm_660_flat = mm_660.reshape(total_rows, hidden_size)
    add_215_flat = add_215.reshape(total_rows, hidden_size)

    fused_rms_norm_backward_add_kernel[grid](
        mm_656_flat, mm_658_flat, mm_660_flat,
        add_1_flat, inv_rms, weight,
        add_215_flat,
        final_result, grad_weight,
        hidden_size,
        stride_input,
        mm_656_flat.stride(0), mm_658_flat.stride(0), mm_660_flat.stride(0),
        add_215_flat.stride(0),
        final_result.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    reshape_default = out1.view(seq_len, hidden_size)
    reshape_default_1 = out2.view(seq_len, hidden_size)
    reshape_default_2 = out3.view(seq_len, hidden_size)
    final_result = final_result.view(add_215.shape)

    return (reshape_default, reshape_default_1, reshape_default_2, final_result, grad_weight)