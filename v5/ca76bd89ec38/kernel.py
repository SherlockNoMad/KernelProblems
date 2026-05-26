import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    x_ptr,           # bf16, shape [8192, H*128]
    idx_ptr,         # int32, shape [8192]
    freqs_real_ptr,  # f32, shape [8192, 64]
    freqs_imag_ptr,  # f32, shape [8192, 64]
    out_ptr,         # bf16, shape [1, H, 8192, 128]
    H: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # program_id(0) = token index (0..SEQ)
    # program_id(1) = head index (0..H)
    tok = tl.program_id(0)
    head = tl.program_id(1)

    # Load freq index for this token
    freq_idx = tl.load(idx_ptr + tok).to(tl.int64)

    # 64 pairs per head
    pair = tl.arange(0, 64)  # 64 pairs

    # Load freqs: arg582_1[freq_idx, pair] -> real and imag (f32)
    freq_off = freq_idx * 64 + pair
    cos_v = tl.load(freqs_real_ptr + freq_off)
    sin_v = tl.load(freqs_imag_ptr + freq_off)

    # Load x: x[tok, head*128 + pair*2], x[tok, head*128 + pair*2 + 1]
    # x stride: row = H*128
    base = tok * (H * 128) + head * 128
    x_real = tl.load(x_ptr + base + pair * 2).to(tl.float32)
    x_imag = tl.load(x_ptr + base + pair * 2 + 1).to(tl.float32)

    # Complex multiply
    out_real = x_real * cos_v - x_imag * sin_v
    out_imag = x_real * sin_v + x_imag * cos_v

    # Store to out: out[0, head, tok, pair*2 : pair*2+2]
    # out shape [1, H, 8192, 128], strides: H -> 8192*128, tok -> 128
    out_base = head * (SEQ * 128) + tok * 128
    tl.store(out_ptr + out_base + pair * 2, out_real.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_base + pair * 2 + 1, out_imag.to(out_ptr.dtype.element_ty))


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    """
    Fused rotary embedding application.
    
    Fuses: index lookup of freqs, complex multiplication, dtype conversion,
    and final transpose layout, all in a single Triton kernel per output.
    Outputs 2,3 are duplicates of 0,1 so we just clone them.
    """
    assert arg586_1.dtype == torch.int32
    assert arg582_1.dtype == torch.complex64
    assert mm_217.dtype == torch.bfloat16
    assert mm_218.dtype == torch.bfloat16
    
    SEQ = 8192
    # Squeeze arg586_1 from [1, 8192] -> [8192]
    idx = arg586_1.view(-1).contiguous()
    
    # Separate complex into real/imag float32 views (view_as_real produces contiguous float32)
    # arg582_1 is [8192, 64] complex64; reinterpret as [8192, 64, 2] float32
    freqs_real_view = torch.view_as_real(arg582_1).contiguous()  # [8192, 64, 2] f32
    freqs_real = freqs_real_view[..., 0].contiguous()  # [8192, 64]
    freqs_imag = freqs_real_view[..., 1].contiguous()
    
    # Output for 32 heads
    out_32 = torch.empty((1, 32, SEQ, 128), dtype=torch.bfloat16, device=mm_217.device)
    out_8 = torch.empty((1, 8, SEQ, 128), dtype=torch.bfloat16, device=mm_218.device)
    
    # Ensure mm_217 / mm_218 contiguous
    mm_217_c = mm_217.contiguous()
    mm_218_c = mm_218.contiguous()
    
    grid_32 = (SEQ, 32)
    _rotary_kernel[grid_32](
        mm_217_c, idx, freqs_real, freqs_imag, out_32,
        H=32, SEQ=SEQ, BLOCK_H=1,
    )
    
    grid_8 = (SEQ, 8)
    _rotary_kernel[grid_8](
        mm_218_c, idx, freqs_real, freqs_imag, out_8,
        H=8, SEQ=SEQ, BLOCK_H=1,
    )
    
    # Outputs 2 and 3 are identical to 0 and 1
    out_32_dup = out_32.clone()
    out_8_dup = out_8.clone()
    
    return (out_32, out_8, out_32_dup, out_8_dup)