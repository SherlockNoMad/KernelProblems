import torch
import triton
import triton.language as tl


@triton.jit
def fused_rms_norm_kernel(
    input_ptr, weight_ptr, output_ptr,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_size

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
    vals_f32 = vals.to(tl.float32)

    variance_sum = tl.sum(vals_f32 * vals_f32)
    rstd = 1.0 / tl.sqrt(variance_sum / hidden_size + eps)

    weights = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    weights_f32 = weights.to(tl.float32)

    normalized = vals_f32 * rstd * weights_f32

    tl.store(output_ptr + row_start + offsets, normalized.to(input_ptr.dtype.element_ty), mask=mask)


@triton.jit
def strided_copy_kernel_vec4(
    input_ptr, output_ptr,
    num_rows, row_len,
    input_stride_row,
    output_stride_row,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    in_ptrs = input_ptr + row_idx * input_stride_row + offsets
    vals = tl.load(in_ptrs, mask=mask, other=0.0)

    out_ptrs = output_ptr + row_idx * output_stride_row + offsets
    tl.store(out_ptrs, vals, mask=mask)


def kernel_function(getitem_1619, add_62, getitem_1620, getitem_1621, getitem_1622):
    device = add_62.device
    dtype = add_62.dtype

    # Weight tensor: small copy [8, 512] with stride [27264000, 1] -> contiguous
    # Small enough that a simple Triton kernel is fine
    weight_contiguous = torch.empty(8, 512, dtype=dtype, device=device)
    strided_copy_kernel_vec4[(8, 1)](
        getitem_1619, weight_contiguous,
        8, 512,
        getitem_1619.stride(0),
        512,
        BLOCK_SIZE=512
    )
    weight = weight_contiguous.reshape(4096)

    # RMS norm - single pass
    batch_size, seq_len, hidden_size = add_62.shape
    N = batch_size * seq_len

    normalized_output = torch.empty_like(add_62)
    input_reshaped = add_62.view(N, hidden_size)
    output_reshaped = normalized_output.view(N, hidden_size)

    fused_rms_norm_kernel[(N,)](
        input_reshaped, weight, output_reshaped,
        hidden_size, 1e-05,
        BLOCK_SIZE=4096
    )

    reshape_default = normalized_output.reshape(8192, 4096)
    reshape_default_1 = normalized_output.reshape(8192, 4096)

    # Large strided copies - use PyTorch's .contiguous() which leverages
    # hardware copy engines / optimized CUDA memcpy, freeing SMs
    # [8, 7340032] stride [27264000, 1] -> contiguous
    w1_contiguous = getitem_1620.contiguous()
    t_default = w1_contiguous.view(14336, 4096).t()

    w2_contiguous = getitem_1621.contiguous()
    t_default_1 = w2_contiguous.view(14336, 4096).t()

    w3_contiguous = getitem_1622.contiguous()
    t_default_2 = w3_contiguous.view(4096, 14336).t()

    return (reshape_default, reshape_default_1, t_default, t_default_1, t_default_2)