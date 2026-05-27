import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel_multi_head(
    idx_ptr,
    rot_ptr,
    mm_ptr,
    out_ptr,
    S,
    H: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one (s, head_group)
    num_head_blocks = H // HEADS_PER_BLOCK
    s = pid // num_head_blocks
    h_block = pid % num_head_blocks

    idx = tl.load(idx_ptr + s).to(tl.int64)

    pair_offs = tl.arange(0, 64)
    rot_base = rot_ptr + idx * 128
    rot_real = tl.load(rot_base + pair_offs * 2)
    rot_imag = tl.load(rot_base + pair_offs * 2 + 1)

    h_start = h_block * HEADS_PER_BLOCK
    head_offs = tl.arange(0, HEADS_PER_BLOCK)  # [HEADS_PER_BLOCK]

    # Compute base pointers for all heads in this block
    # mm row: s*(H*128), each head at h*128
    mm_row_base = mm_ptr + s * (H * 128)
    out_row_base = out_ptr + s * (H * 128)

    # Build 2D offsets: [HEADS_PER_BLOCK, 64]
    h_idx = h_start + head_offs[:, None]  # [HEADS_PER_BLOCK, 1]
    p_idx = pair_offs[None, :]            # [1, 64]

    real_offs = h_idx * 128 + p_idx * 2
    imag_offs = h_idx * 128 + p_idx * 2 + 1

    x_real = tl.load(mm_row_base + real_offs).to(tl.float32)
    x_imag = tl.load(mm_row_base + imag_offs).to(tl.float32)

    rr = rot_real[None, :]
    ri = rot_imag[None, :]

    out_real = x_real * rr - x_imag * ri
    out_imag = x_real * ri + x_imag * rr

    tl.store(out_row_base + real_offs, out_real.to(tl.bfloat16))
    tl.store(out_row_base + imag_offs, out_imag.to(tl.bfloat16))


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    assert arg586_1.is_cuda and arg582_1.is_cuda and mm_217.is_cuda and mm_218.is_cuda
    assert arg582_1.dtype == torch.complex64
    assert mm_217.dtype == torch.bfloat16
    assert mm_218.dtype == torch.bfloat16

    idx = arg586_1.squeeze(0).contiguous().to(torch.int32)
    S = idx.shape[0]

    rot_f32 = torch.view_as_real(arg582_1).contiguous()

    out_xq = torch.empty((1, S, 32, 128), dtype=torch.bfloat16, device=mm_217.device)
    out_xk = torch.empty((1, S, 8, 128), dtype=torch.bfloat16, device=mm_218.device)

    mm_217_c = mm_217.contiguous()
    mm_218_c = mm_218.contiguous()

    # xq: H=32, group 8 heads per block -> 4 blocks per s
    HEADS_Q = 8
    grid_q = (S * (32 // HEADS_Q),)
    _rotary_kernel_multi_head[grid_q](
        idx, rot_f32, mm_217_c, out_xq,
        S, H=32, HEADS_PER_BLOCK=HEADS_Q,
        num_warps=4, num_stages=2,
    )

    # xk: H=8, group all 8 heads per block -> 1 block per s
    HEADS_K = 8
    grid_k = (S * (8 // HEADS_K),)
    _rotary_kernel_multi_head[grid_k](
        idx, rot_f32, mm_218_c, out_xk,
        S, H=8, HEADS_PER_BLOCK=HEADS_K,
        num_warps=4, num_stages=2,
    )

    return (out_xq, out_xk)