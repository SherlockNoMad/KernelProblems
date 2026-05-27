import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel_v3(
    mm_ptr,
    idx_ids_ptr,
    cos_table_ptr,
    sin_table_ptr,
    out1_ptr,
    out2_ptr,
    S,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    d_offs = tl.arange(0, D)

    idx_ids = tl.load(idx_ids_ptr + s_offs, mask=s_mask, other=0)

    cos_v = tl.load(cos_table_ptr + idx_ids[:, None] * D + d_offs[None, :],
                    mask=s_mask[:, None], other=0.0)
    sin_v = tl.load(sin_table_ptr + idx_ids[:, None] * D + d_offs[None, :],
                    mask=s_mask[:, None], other=0.0)

    row_stride = H * D * 2
    head_off = pid_h * (D * 2)

    re_off = s_offs[:, None] * row_stride + head_off + d_offs[None, :] * 2
    im_off = re_off + 1
    mask2d = s_mask[:, None]

    a_re = tl.load(mm_ptr + re_off, mask=mask2d, other=0.0).to(tl.float32)
    a_im = tl.load(mm_ptr + im_off, mask=mask2d, other=0.0).to(tl.float32)

    out_re = (a_re * cos_v - a_im * sin_v).to(tl.bfloat16)
    out_im = (a_re * sin_v + a_im * cos_v).to(tl.bfloat16)

    tl.store(out1_ptr + re_off, out_re, mask=mask2d)
    tl.store(out1_ptr + im_off, out_im, mask=mask2d)
    tl.store(out2_ptr + re_off, out_re, mask=mask2d)
    tl.store(out2_ptr + im_off, out_im, mask=mask2d)


def kernel_function(arg586_1, arg582_1, mm_217, mm_218):
    S = 8192
    D = 64
    H1 = 32
    H2 = 8

    squeeze_dim = arg586_1.view(-1).to(torch.int32).contiguous()

    arg582_real = arg582_1.real.contiguous()
    arg582_imag = arg582_1.imag.contiguous()

    mm_217_c = mm_217.contiguous()
    mm_218_c = mm_218.contiguous()

    out1 = torch.empty((1, S, H1, 128), dtype=torch.bfloat16, device=mm_217.device)
    out3 = torch.empty((1, S, H1, 128), dtype=torch.bfloat16, device=mm_217.device)
    out2 = torch.empty((1, S, H2, 128), dtype=torch.bfloat16, device=mm_218.device)
    out4 = torch.empty((1, S, H2, 128), dtype=torch.bfloat16, device=mm_218.device)

    BLOCK_S = 32
    grid1 = (triton.cdiv(S, BLOCK_S), H1)
    _rope_kernel_v3[grid1](
        mm_217_c, squeeze_dim, arg582_real, arg582_imag, out1, out3,
        S, BLOCK_S=BLOCK_S, D=D, H=H1,
        num_warps=4, num_stages=4,
    )

    grid2 = (triton.cdiv(S, BLOCK_S), H2)
    _rope_kernel_v3[grid2](
        mm_218_c, squeeze_dim, arg582_real, arg582_imag, out2, out4,
        S, BLOCK_S=BLOCK_S, D=D, H=H2,
        num_warps=4, num_stages=4,
    )

    return (out1, out2, out3, out4)