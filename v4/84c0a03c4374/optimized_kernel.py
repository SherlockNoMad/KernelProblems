import torch
import triton
import triton.language as tl


@triton.jit
def copy_strided_to_contiguous_kernel(
    src_ptr, dst_ptr,
    nrows, ncols, src_stride_0, src_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nrows * ncols
    row = offsets // ncols
    col = offsets % ncols
    src_idx = row * src_stride_0 + col * src_stride_1
    val = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, val, mask=mask)


@triton.jit
def fused_rms_norm_fwd_kernel(
    input_ptr, weight_ptr,
    output_ptr, inv_rms_ptr,
    seq_len, hidden_dim,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= seq_len:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    x = tl.load(input_ptr + row_idx * input_stride_0 + offsets * input_stride_1,
                mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / hidden_dim
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    x_normed = x * inv_rms
    output = x_normed * weight
    tl.store(output_ptr + row_idx * output_stride_0 + offsets * output_stride_1,
             output.to(tl.bfloat16), mask=mask)
    tl.store(inv_rms_ptr + row_idx, inv_rms)


@triton.jit
def fused_rms_backward_add_atomic_kernel(
    mm_230_ptr, mm_232_ptr,
    input_ptr, weight_ptr, inv_rms_ptr,
    getitem_420_ptr,
    grad_input_out_ptr,
    grad_weight_ptr,
    seq_len, hidden_dim: tl.constexpr,
    mm_stride_0, mm_stride_1,
    input_stride_0, input_stride_1,
    getitem_stride_0, getitem_stride_1,
    out_stride_0, out_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= seq_len:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_bf16 = tl.load(mm_230_ptr + row_idx * mm_stride_0 + offsets * mm_stride_1,
                     mask=mask, other=0.0)
    b_bf16 = tl.load(mm_232_ptr + row_idx * mm_stride_0 + offsets * mm_stride_1,
                     mask=mask, other=0.0)
    grad_out_bf16 = a_bf16 + b_bf16
    grad_out = grad_out_bf16.to(tl.float32)

    x = tl.load(input_ptr + row_idx * input_stride_0 + offsets * input_stride_1,
                mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(inv_rms_ptr + row_idx).to(tl.float32)

    x_hat = x * inv_rms
    grad_weight_local = grad_out * x_hat

    grad_normed = grad_out * weight
    c = tl.sum(grad_normed * x_hat, axis=0) / hidden_dim
    grad_input = (grad_normed - x_hat * c) * inv_rms

    g420 = tl.load(getitem_420_ptr + row_idx * getitem_stride_0 + offsets * getitem_stride_1,
                   mask=mask, other=0.0).to(tl.float32)
    result = g420 + grad_input

    tl.store(grad_input_out_ptr + row_idx * out_stride_0 + offsets * out_stride_1,
             result.to(tl.bfloat16), mask=mask)

    tl.atomic_add(grad_weight_ptr + offsets, grad_weight_local, mask=mask)


def kernel_function(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420):
    device = add_62_recomputed.device
    batch_size, seq_len_total, hidden_dim = add_62_recomputed.shape
    total_rows = batch_size * seq_len_total

    weight_viewed = getitem_1624
    nrows_w = weight_viewed.shape[0]
    ncols_w = weight_viewed.shape[1]
    num_weight_elements = nrows_w * ncols_w

    weight_cloned = torch.empty(num_weight_elements, dtype=torch.bfloat16, device=device)
    CLONE_BLOCK = 1024
    clone_grid = (triton.cdiv(num_weight_elements, CLONE_BLOCK),)
    copy_strided_to_contiguous_kernel[clone_grid](
        weight_viewed, weight_cloned,
        nrows_w, ncols_w,
        weight_viewed.stride(0), weight_viewed.stride(1),
        BLOCK_SIZE=CLONE_BLOCK,
    )
    weight_processed = weight_cloned

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    input_flattened = add_62_recomputed.view(-1, hidden_dim)
    normed_output = torch.empty([total_rows, hidden_dim], dtype=torch.bfloat16, device=device)
    inv_rms = torch.empty([total_rows], dtype=torch.float32, device=device)

    fused_rms_norm_fwd_kernel[(total_rows,)](
        input_flattened, weight_processed,
        normed_output, inv_rms,
        total_rows, hidden_dim,
        input_flattened.stride(0), input_flattened.stride(1),
        normed_output.stride(0), normed_output.stride(1),
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    reshape_default = normed_output.view(total_rows, hidden_dim)
    reshape_default_1 = normed_output.view(total_rows, hidden_dim)

    add_tensor_1 = torch.empty([batch_size, seq_len_total, hidden_dim], dtype=torch.bfloat16, device=device)
    grad_weight_final = torch.zeros([hidden_dim], dtype=torch.float32, device=device)

    getitem_420_flat = getitem_420.view(-1, hidden_dim)
    add_tensor_1_flat = add_tensor_1.view(-1, hidden_dim)

    fused_rms_backward_add_atomic_kernel[(total_rows,)](
        mm_230, mm_232,
        input_flattened, weight_processed, inv_rms,
        getitem_420_flat,
        add_tensor_1_flat,
        grad_weight_final,
        total_rows, hidden_dim,
        mm_230.stride(0), mm_230.stride(1),
        input_flattened.stride(0), input_flattened.stride(1),
        getitem_420_flat.stride(0), getitem_420_flat.stride(1),
        add_tensor_1_flat.stride(0), add_tensor_1_flat.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (reshape_default, reshape_default_1, add_tensor_1, grad_weight_final)