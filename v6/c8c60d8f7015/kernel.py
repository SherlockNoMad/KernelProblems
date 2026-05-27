import torch
import triton
import triton.language as tl

# Fused kernel: applies rotary embedding via complex multiplication
# Stages fused:
#   1. Gather rotary embedding row from arg582 using index from arg586 (squeezed)
#   2. Reinterpret mm input (bf16) as float32 pairs (real, imag)
#   3. Complex multiply with rotary
#   4. Cast back to bf16 and store
#
# arg582 is complex64 with shape [8192, 64]; stored as interleaved (real, imag) float32 -> 128 floats per row
# mm_217: [8192, 4096] bf16, viewed as [1, 8192, 32, 64, 2] (head_dim/2=64 pairs)
# Output xq: [1, 8192, 32, 128] bf16
# mm_218: [8192, 1024] bf16, viewed as [1, 8192, 8, 64, 2]
# Output xk: [1, 8192, 8, 128] bf16


@triton.jit
def _rotary_kernel(
    idx_ptr,         # int32, [S]
    rot_ptr,         # float32, [N_ROT, 64, 2] (treated as float, complex64 reinterpreted)
    mm_ptr,          # bf16, [S, H*64*2] where H = num_heads
    out_ptr,         # bf16, [S, H, 128]
    S,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # program_id(0) = seq position, program_id(1) = head block
    s = tl.program_id(0)
    h_block = tl.program_id(1)

    if s >= S:
        return

    # Load index
    idx = tl.load(idx_ptr + s).to(tl.int64)

    # 64 pairs per head, each pair is (real, imag)
    # offsets for pairs
    pair_offs = tl.arange(0, 64)  # 0..63

    # Load rotary row: real + imag interleaved at rot_ptr + idx*128 + 2*pair_offs
    rot_base = rot_ptr + idx * 128
    rot_real = tl.load(rot_base + pair_offs * 2)
    rot_imag = tl.load(rot_base + pair_offs * 2 + 1)

    # Iterate heads in this block
    h_start = h_block * BLOCK_H
    for h_off in tl.static_range(0, BLOCK_H):
        h = h_start + h_off
        if h < H:
            # mm input row: s, then head h, 64 pairs
            mm_base = mm_ptr + s * (H * 128) + h * 128
            x_real_bf16 = tl.load(mm_base + pair_offs * 2)
            x_imag_bf16 = tl.load(mm_base + pair_offs * 2 + 1)
            x_real = x_real_bf16.to(tl.float32)
            x_imag = x_imag_bf16.to(tl.float32)

            # Complex multiply: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
            out_real = x_real * rot_real - x_imag * rot_imag
            out_imag = x_real * rot_imag + x_imag * rot_real

            out_base = out_ptr + s * (H * 128) + h * 128
            tl.store(out_base + pair_offs * 2, out_real.to(tl.bfloat16))
            tl.store(out_base + pair_offs * 2 + 1, out_imag.to(tl.bfloat16))


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    """
    Fused rotary embedding application.

    Fused stages:
      - Squeeze arg586 and gather corresponding rotary rows from arg582
      - View mm tensors as complex pairs, multiply by rotary, cast back to bf16
    """
    assert arg586_1.is_cuda and arg582_1.is_cuda and mm_217.is_cuda and mm_218.is_cuda
    assert arg582_1.dtype == torch.complex64
    assert mm_217.dtype == torch.bfloat16
    assert mm_218.dtype == torch.bfloat16

    # Squeeze arg586_1 along dim 0 -> [8192]
    idx = arg586_1.squeeze(0).contiguous().to(torch.int32)
    S = idx.shape[0]  # 8192

    # Reinterpret complex64 rotary as float32 with last dim *2
    # arg582_1: [8192, 64] complex64 -> [8192, 64, 2] float32
    rot_f32 = torch.view_as_real(arg582_1).contiguous()  # [N_ROT, 64, 2] float32

    # Outputs
    out_xq = torch.empty((1, S, 32, 128), dtype=torch.bfloat16, device=mm_217.device)
    out_xk = torch.empty((1, S, 8, 128), dtype=torch.bfloat16, device=mm_218.device)

    mm_217_c = mm_217.contiguous()
    mm_218_c = mm_218.contiguous()

    # Launch for xq (H=32)
    BLOCK_H_Q = 4
    grid_q = (S, triton.cdiv(32, BLOCK_H_Q))
    _rotary_kernel[grid_q](
        idx, rot_f32, mm_217_c, out_xq,
        S, H=32, BLOCK_H=BLOCK_H_Q,
    )

    # Launch for xk (H=8)
    BLOCK_H_K = 4
    grid_k = (S, triton.cdiv(8, BLOCK_H_K))
    _rotary_kernel[grid_k](
        idx, rot_f32, mm_218_c, out_xk,
        S, H=8, BLOCK_H=BLOCK_H_K,
    )

    return (out_xq, out_xk)