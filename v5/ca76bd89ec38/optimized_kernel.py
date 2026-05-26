import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel_v2(
    x_ptr,
    idx_ptr,
    freqs_ptr,
    out_ptr,
    H: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    tok_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    head_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d128 = tl.arange(0, 128)

    tok_mask = tok_offs < SEQ
    head_mask = head_offs < H

    freq_idx = tl.load(idx_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int64)

    # Contiguous freq load [BLOCK_T, 128]
    freq_base = freq_idx[:, None] * 128 + d128[None, :]
    freqs_flat = tl.load(freqs_ptr + freq_base, mask=tok_mask[:, None], other=0.0)
    freqs_r = tl.reshape(freqs_flat, (BLOCK_T, 64, 2))
    e0 = tl.reshape(tl.cast(tl.arange(0, 2) == 0, tl.float32), (1, 1, 2))
    e1 = tl.reshape(tl.cast(tl.arange(0, 2) == 1, tl.float32), (1, 1, 2))
    cos_v = tl.reshape(tl.sum(freqs_r * e0, axis=2), (BLOCK_T, 64))
    sin_v = tl.reshape(tl.sum(freqs_r * e1, axis=2), (BLOCK_T, 64))

    # Contiguous x load [BLOCK_T, BLOCK_H, 128]
    x_base = (
        tok_offs[:, None, None] * (H * 128)
        + head_offs[None, :, None] * 128
        + d128[None, None, :]
    )
    mask3 = tok_mask[:, None, None] & head_mask[None, :, None]
    x_flat = tl.load(x_ptr + x_base, mask=mask3, other=0.0).to(tl.float32)
    x_r = tl.reshape(x_flat, (BLOCK_T, BLOCK_H, 64, 2))
    e0b = tl.reshape(tl.cast(tl.arange(0, 2) == 0, tl.float32), (1, 1, 1, 2))
    e1b = tl.reshape(tl.cast(tl.arange(0, 2) == 1, tl.float32), (1, 1, 1, 2))
    x_real = tl.reshape(tl.sum(x_r * e0b, axis=3), (BLOCK_T, BLOCK_H, 64))
    x_imag = tl.reshape(tl.sum(x_r * e1b, axis=3), (BLOCK_T, BLOCK_H, 64))

    cos_b = cos_v[:, None, :]
    sin_b = sin_v[:, None, :]

    out_real = x_real * cos_b - x_imag * sin_b
    out_imag = x_real * sin_b + x_imag * cos_b

    out_stacked = tl.reshape(
        tl.join(out_real, out_imag),
        (BLOCK_T, BLOCK_H, 128),
    )

    out_base = (
        head_offs[None, :, None] * (SEQ * 128)
        + tok_offs[:, None, None] * 128
        + d128[None, None, :]
    )
    tl.store(out_ptr + out_base, out_stacked.to(out_ptr.dtype.element_ty), mask=mask3)


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    SEQ = 8192
    idx = arg586_1.view(-1).contiguous()

    freqs_f32 = torch.view_as_real(arg582_1).contiguous()

    out_32 = torch.empty((1, 32, SEQ, 128), dtype=torch.bfloat16, device=mm_217.device)
    out_8 = torch.empty((1, 8, SEQ, 128), dtype=torch.bfloat16, device=mm_218.device)

    mm_217_c = mm_217.contiguous()
    mm_218_c = mm_218.contiguous()

    BLOCK_T_32 = 32
    BLOCK_H_32 = 8
    grid_32 = (SEQ // BLOCK_T_32, 32 // BLOCK_H_32)
    _rotary_kernel_v2[grid_32](
        mm_217_c, idx, freqs_f32, out_32,
        H=32, SEQ=SEQ, BLOCK_T=BLOCK_T_32, BLOCK_H=BLOCK_H_32,
        num_warps=8, num_stages=2,
    )

    BLOCK_T_8 = 32
    BLOCK_H_8 = 8
    grid_8 = (SEQ // BLOCK_T_8, 8 // BLOCK_H_8)
    _rotary_kernel_v2[grid_8](
        mm_218_c, idx, freqs_f32, out_8,
        H=8, SEQ=SEQ, BLOCK_T=BLOCK_T_8, BLOCK_H=BLOCK_H_8,
        num_warps=8, num_stages=2,
    )

    out_32_dup = out_32
    out_8_dup = out_8

    return (out_32, out_8, out_32_dup, out_8_dup)