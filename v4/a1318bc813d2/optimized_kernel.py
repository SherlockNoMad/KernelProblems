import torch
import triton
import triton.language as tl


@triton.jit
def fused_rotary_multi_seq_kernel(
    input_ptr, rotary_real_ptr, rotary_imag_ptr, output_ptr,
    seq_len, num_heads, half_head_dim,
    input_stride_head, input_stride_seq,
    rotary_stride_seq,
    output_stride_seq,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    seq_base = pid_seq * BLOCK_SEQ
    seq_offsets = tl.arange(0, BLOCK_SEQ)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < half_head_dim

    real_offsets = dim_offsets * 2
    imag_offsets = dim_offsets * 2 + 1

    inp_head_base = pid_head * input_stride_head
    out_head_base = pid_head * half_head_dim * 2

    for s in range(BLOCK_SEQ):
        seq_idx = seq_base + s
        if seq_idx < seq_len:
            rot_base = seq_idx * rotary_stride_seq
            rot_r = tl.load(rotary_real_ptr + rot_base + dim_offsets, mask=dim_mask, other=1.0)
            rot_i = tl.load(rotary_imag_ptr + rot_base + dim_offsets, mask=dim_mask, other=0.0)

            inp_base = inp_head_base + seq_idx * input_stride_seq
            inp_real = tl.load(input_ptr + inp_base + real_offsets, mask=dim_mask, other=0.0).to(tl.float32)
            inp_imag = tl.load(input_ptr + inp_base + imag_offsets, mask=dim_mask, other=0.0).to(tl.float32)

            out_real = inp_real * rot_r - inp_imag * rot_i
            out_imag = inp_real * rot_i + inp_imag * rot_r

            out_base = seq_idx * output_stride_seq + out_head_base
            tl.store(output_ptr + out_base + real_offsets, out_real.to(tl.bfloat16), mask=dim_mask)
            tl.store(output_ptr + out_base + imag_offsets, out_imag.to(tl.bfloat16), mask=dim_mask)


@triton.jit
def transpose_v_multi_seq_kernel(
    input_ptr, output_ptr,
    seq_len, num_heads, head_dim,
    input_stride_head, input_stride_seq,
    output_stride_seq,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    seq_base = pid_seq * BLOCK_SEQ
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < head_dim

    inp_head_base = pid_head * input_stride_head
    out_head_base = pid_head * head_dim

    for s in range(BLOCK_SEQ):
        seq_idx = seq_base + s
        if seq_idx < seq_len:
            inp_base = inp_head_base + seq_idx * input_stride_seq
            data = tl.load(input_ptr + inp_base + dim_offsets, mask=dim_mask, other=0.0)
            out_base = seq_idx * output_stride_seq + out_head_base
            tl.store(output_ptr + out_base + dim_offsets, data, mask=dim_mask)


def kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643):
    device = _conj_62.device
    seq_len = 8192

    conj_62_view = torch.view_as_real(_conj_62)
    conj_62_real = conj_62_view[..., 0].contiguous()
    conj_62_imag = conj_62_view[..., 1].contiguous()

    conj_63_view = torch.view_as_real(_conj_63)
    conj_63_real = conj_63_view[..., 0].contiguous()
    conj_63_imag = conj_63_view[..., 1].contiguous()

    BLOCK_SEQ_Q = 8
    BLOCK_SEQ_K = 16
    BLOCK_SEQ_V = 16

    q_output = torch.empty([seq_len, 32 * 128], dtype=torch.bfloat16, device=device)
    grid_q = (seq_len // BLOCK_SEQ_Q, 32)
    fused_rotary_multi_seq_kernel[grid_q](
        getitem_641, conj_63_real, conj_63_imag, q_output,
        seq_len, 32, 64,
        128, 4096,
        64,
        4096,
        BLOCK_SEQ=BLOCK_SEQ_Q,
        BLOCK_DIM=64,
    )
    q_final = q_output.t()

    k_output = torch.empty([seq_len, 8 * 128], dtype=torch.bfloat16, device=device)
    grid_k = (seq_len // BLOCK_SEQ_K, 8)
    fused_rotary_multi_seq_kernel[grid_k](
        getitem_642, conj_62_real, conj_62_imag, k_output,
        seq_len, 8, 64,
        128, 1024,
        64,
        1024,
        BLOCK_SEQ=BLOCK_SEQ_K,
        BLOCK_DIM=64,
    )
    k_final = k_output.t()

    v_output = torch.empty([seq_len, 8 * 128], dtype=torch.bfloat16, device=device)
    grid_v = (seq_len // BLOCK_SEQ_V, 8)
    transpose_v_multi_seq_kernel[grid_v](
        getitem_643, v_output,
        seq_len, 8, 128,
        128, 1024,
        1024,
        BLOCK_SEQ=BLOCK_SEQ_V,
        BLOCK_DIM=128,
    )
    v_final = v_output.t()

    return (q_final, k_final, v_final)