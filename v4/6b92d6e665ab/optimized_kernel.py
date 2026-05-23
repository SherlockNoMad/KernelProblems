import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_kernel(
    pos_ptr,
    rope_ptr,
    q_in_ptr,
    k_in_ptr,
    q_out1_ptr,
    k_out1_ptr,
    q_out2_ptr,
    k_out2_ptr,
    seq_len,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    HALF_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    is_k = pid_head >= q_heads
    head_idx = pid_head - q_heads * is_k.to(tl.int32)

    seq_off = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_off < seq_len

    pos = tl.load(pos_ptr + seq_off, mask=seq_mask, other=0)

    dim_off = tl.arange(0, HALF_DIM)

    rope_real = tl.load(rope_ptr + pos[:, None] * (HALF_DIM * 2) + dim_off[None, :] * 2,
                        mask=seq_mask[:, None], other=0.0)
    rope_imag = tl.load(rope_ptr + pos[:, None] * (HALF_DIM * 2) + dim_off[None, :] * 2 + 1,
                        mask=seq_mask[:, None], other=0.0)

    total_heads_q = q_heads
    total_heads_k = k_heads

    in_ptr = tl.where(is_k, k_in_ptr, q_in_ptr)
    num_h = tl.where(is_k, total_heads_k, total_heads_q)

    in_seq_stride = tl.where(is_k, total_heads_k * HEAD_DIM, total_heads_q * HEAD_DIM)
    in_base_offset = seq_off[:, None] * in_seq_stride + head_idx * HEAD_DIM

    in_real = tl.load(in_ptr + in_base_offset + dim_off[None, :] * 2,
                      mask=seq_mask[:, None], other=0.0).to(tl.float32)
    in_imag = tl.load(in_ptr + in_base_offset + dim_off[None, :] * 2 + 1,
                      mask=seq_mask[:, None], other=0.0).to(tl.float32)

    out_real = in_real * rope_real - in_imag * rope_imag
    out_imag = in_real * rope_imag + in_imag * rope_real

    out_real_bf = out_real.to(tl.bfloat16)
    out_imag_bf = out_imag.to(tl.bfloat16)

    out_head_stride = seq_len * HEAD_DIM
    out1_ptr = tl.where(is_k, k_out1_ptr, q_out1_ptr)
    out2_ptr = tl.where(is_k, k_out2_ptr, q_out2_ptr)

    out_base = head_idx * out_head_stride + seq_off[:, None] * HEAD_DIM

    tl.store(out1_ptr + out_base + dim_off[None, :] * 2, out_real_bf, mask=seq_mask[:, None])
    tl.store(out1_ptr + out_base + dim_off[None, :] * 2 + 1, out_imag_bf, mask=seq_mask[:, None])
    tl.store(out2_ptr + out_base + dim_off[None, :] * 2, out_real_bf, mask=seq_mask[:, None])
    tl.store(out2_ptr + out_base + dim_off[None, :] * 2 + 1, out_imag_bf, mask=seq_mask[:, None])


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    device = arg586_1.device
    seq_len = 8192
    q_heads = 32
    k_heads = 8
    head_dim = 128
    half_dim = 64

    pos_indices = arg586_1.reshape(-1)
    rope_f32 = torch.view_as_real(arg582_1).contiguous()

    q_out1 = torch.empty((1, q_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    k_out1 = torch.empty((1, k_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    q_out2 = torch.empty((1, q_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    k_out2 = torch.empty((1, k_heads, seq_len, head_dim), dtype=torch.bfloat16, device=device)

    BLOCK_SEQ = 32
    total_heads = q_heads + k_heads
    grid = (triton.cdiv(seq_len, BLOCK_SEQ), total_heads)

    fused_rope_kernel[grid](
        pos_indices, rope_f32,
        mm_217, mm_218,
        q_out1, k_out1, q_out2, k_out2,
        seq_len, q_heads, k_heads, half_dim, head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=4, num_stages=2,
    )

    return (q_out1, k_out1, q_out2, k_out2)