import torch
import triton
import triton.language as tl


@triton.jit
def _rope_complex_mul_kernel(
    x_ptr,
    freq_ptr,
    out_ptr,
    n_pairs,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * VEC
    offs = base + tl.arange(0, BLOCK * VEC)
    mask = offs < n_pairs

    d = offs % D
    tmp = offs // D
    head = tmp % H
    seq = tmp // H

    x_pair_idx = seq * (H * D) + head * D + d
    x_pair = tl.load(x_ptr + x_pair_idx, mask=mask, other=0)
    x_pair_u32 = x_pair.to(tl.uint32, bitcast=True)
    a_u16 = (x_pair_u32 & 0xFFFF).to(tl.uint16)
    b_u16 = (x_pair_u32 >> 16).to(tl.uint16)
    a = a_u16.to(tl.bfloat16, bitcast=True).to(tl.float32)
    b = b_u16.to(tl.bfloat16, bitcast=True).to(tl.float32)

    f_pair_idx = seq * D + d
    f_pair = tl.load(freq_ptr + f_pair_idx, mask=mask, other=0)
    f_pair_u64 = f_pair.to(tl.uint64, bitcast=True)
    c_u32 = (f_pair_u64 & 0xFFFFFFFF).to(tl.uint32)
    e_u32 = (f_pair_u64 >> 32).to(tl.uint32)
    c = c_u32.to(tl.float32, bitcast=True)
    e = e_u32.to(tl.float32, bitcast=True)

    real = a * c - b * e
    imag = a * e + b * c

    r_u16 = real.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    i_u16 = imag.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    out_u32 = r_u16 | (i_u16 << 16)

    out_pair_idx = seq * (H * D) + head * D + d
    tl.store(out_ptr + out_pair_idx, out_u32, mask=mask)


def kernel_function(mm_218, view_788):
    mm_218_c = mm_218.contiguous()
    view_788_c = view_788.contiguous()

    x_pairs = mm_218_c.view(torch.int32)
    freq_pairs = view_788_c.view(torch.int64)

    out = torch.empty((1, 8192, 8, 128), dtype=torch.bfloat16, device=mm_218.device)
    out_pairs = out.view(torch.int32)

    H = 8
    D = 64
    n_pairs = 8192 * H * D

    BLOCK = 512
    VEC = 4
    grid = (triton.cdiv(n_pairs, BLOCK * VEC),)

    _rope_complex_mul_kernel[grid](
        x_pairs, freq_pairs, out_pairs,
        n_pairs,
        H=H, D=D, BLOCK=BLOCK, VEC=VEC,
        num_warps=4,
    )
    return out