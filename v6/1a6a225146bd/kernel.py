import torch
import triton
import triton.language as tl


# Fused kernel that performs:
# 1. Index into arg582_1 (complex64, shape [8192, 64]) via squeeze_dim from arg586_1
# 2. Reshape mm_217/mm_218 to complex pairs, multiply by indexed complex tensor (broadcasting)
# 3. Cast back to bfloat16
# All in one kernel per output (two outputs share the index tensor)
#
# Layout:
#   mm_217: [8192, 4096] bf16 -> view [1, 8192, 32, 64, 2] (complex pairs)
#   mm_218: [8192, 1024] bf16 -> view [1, 8192, 8, 64, 2] (complex pairs)
#   index:  [1, 8192, 1, 64] complex (real/imag)
#
# For each position (s, h, d) where s in [0,8192), h in heads, d in [0,64):
#   a_re, a_im = mm[s, h*128 + 2*d], mm[s, h*128 + 2*d+1]
#   b_re, b_im = idx[s, d].real, idx[s, d].imag
#   out_re = a_re*b_re - a_im*b_im
#   out_im = a_re*b_im + a_im*b_re
#   out[s, h, 2*d] = out_re; out[s, h, 2*d+1] = out_im


@triton.jit
def _rope_kernel(
    mm_ptr,           # bf16, shape [S, H*128]
    idx_ids_ptr,      # int32, shape [S] (squeeze_dim)
    cos_table_ptr,    # float32 (real part of complex64 table), shape [N, 64]
    sin_table_ptr,    # float32 (imag part), shape [N, 64]
    out_ptr,          # bf16, shape [S, H*128]
    S, H,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,  # 64
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    s = pid_s
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    d_offs = tl.arange(0, D)  # [D]

    # Load index id for this row
    idx_id = tl.load(idx_ids_ptr + s)

    # Load cos/sin (real/imag of complex table) for this row's index
    # shape [D]
    cos_v = tl.load(cos_table_ptr + idx_id * D + d_offs)
    sin_v = tl.load(sin_table_ptr + idx_id * D + d_offs)

    # For each h in BLOCK_H, load real and imag parts
    # mm layout: [S, H, D, 2] flattened => offset = s * (H*D*2) + h * (D*2) + d*2 + (0 or 1)
    # h_offs shape [BLOCK_H], d_offs shape [D]
    base = s * (H * D * 2)

    # 2D offsets [BLOCK_H, D]
    h_b = h_offs[:, None]
    d_b = d_offs[None, :]

    re_off = base + h_b * (D * 2) + d_b * 2
    im_off = re_off + 1

    mask2d = h_mask[:, None] & (d_b < D)

    a_re = tl.load(mm_ptr + re_off, mask=mask2d, other=0.0).to(tl.float32)
    a_im = tl.load(mm_ptr + im_off, mask=mask2d, other=0.0).to(tl.float32)

    # Broadcast cos/sin to [BLOCK_H, D]
    c = cos_v[None, :]
    sn = sin_v[None, :]

    out_re = a_re * c - a_im * sn
    out_im = a_re * sn + a_im * c

    tl.store(out_ptr + re_off, out_re.to(tl.bfloat16), mask=mask2d)
    tl.store(out_ptr + im_off, out_im.to(tl.bfloat16), mask=mask2d)


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    """
    Fused RoPE-style complex multiplication.
    
    Fuses: squeeze + index + view + to_float32 + view_as_complex + complex mul +
           view_as_real + view + to_bfloat16 for both mm_217 and mm_218,
           computed twice (returning 4 outputs, with pairs being identical).
    """
    assert arg586_1.is_cuda and arg582_1.is_cuda
    assert mm_217.is_cuda and mm_218.is_cuda

    S = 8192
    D = 64  # complex dim
    H1 = 32  # heads for mm_217 (4096 / 128)
    H2 = 8   # heads for mm_218 (1024 / 128)

    # squeeze_dim: arg586_1 is [1, 8192], squeeze gives [8192]
    squeeze_dim = arg586_1.view(-1).to(torch.int32).contiguous()
    assert squeeze_dim.numel() == S

    # arg582_1 is complex64 [8192, 64]. Convert to real/imag float32 tables.
    # view_as_real gives [8192, 64, 2] float32; we need contiguous separate parts.
    # Use real() and imag() for clarity (allocation only, no compute on data semantics)
    arg582_real = arg582_1.real.contiguous()  # [8192, 64] float32
    arg582_imag = arg582_1.imag.contiguous()  # [8192, 64] float32

    mm_217_c = mm_217.contiguous()
    mm_218_c = mm_218.contiguous()

    out1 = torch.empty((1, S, H1, 128), dtype=torch.bfloat16, device=mm_217.device)
    out2 = torch.empty((1, S, H2, 128), dtype=torch.bfloat16, device=mm_218.device)

    BLOCK_H1 = 8
    grid1 = (S, triton.cdiv(H1, BLOCK_H1))
    _rope_kernel[grid1](
        mm_217_c, squeeze_dim, arg582_real, arg582_imag, out1,
        S, H1, BLOCK_H=BLOCK_H1, D=D,
    )

    BLOCK_H2 = 8
    grid2 = (S, triton.cdiv(H2, BLOCK_H2))
    _rope_kernel[grid2](
        mm_218_c, squeeze_dim, arg582_real, arg582_imag, out2,
        S, H2, BLOCK_H=BLOCK_H2, D=D,
    )

    # Outputs 3 and 4 are recomputed but identical to 1 and 2 — return clones
    out3 = out1.clone()
    out4 = out2.clone()

    return (out1, out2, out3, out4)