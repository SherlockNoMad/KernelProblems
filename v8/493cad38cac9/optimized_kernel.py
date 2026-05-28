import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    inp_ptr,
    freq_ptr,
    out_ptr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    head_block = tl.program_id(1)

    N_PAIRS: tl.constexpr = HEAD_DIM // 2

    h = head_block * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    h_mask = h < N_HEADS

    two_p = tl.arange(0, HEAD_DIM)
    freq_base = row * HEAD_DIM
    f_pair = tl.load(freq_ptr + freq_base + two_p)
    f_pair_2d = tl.reshape(f_pair, (N_PAIRS, 2))
    f_real, f_imag = tl.split(f_pair_2d)

    row_base = row * (N_HEADS * HEAD_DIM)
    offs = row_base + h[:, None] * HEAD_DIM + two_p[None, :]

    x = tl.load(inp_ptr + offs, mask=h_mask[:, None], other=0.0).to(tl.float32)
    x_3d = tl.reshape(x, (BLOCK_HEADS, N_PAIRS, 2))
    x_real, x_imag = tl.split(x_3d)

    fr = f_real[None, :]
    fi = f_imag[None, :]

    out_real = x_real * fr - x_imag * fi
    out_imag = x_real * fi + x_imag * fr

    out_stacked = tl.join(out_real, out_imag)
    out_flat = tl.reshape(out_stacked, (BLOCK_HEADS, HEAD_DIM))

    tl.store(out_ptr + offs, out_flat.to(tl.bfloat16), mask=h_mask[:, None])


def kernel_function(mm_217, view_788):
    assert mm_217.is_cuda and view_788.is_cuda
    assert mm_217.dtype == torch.bfloat16
    assert view_788.dtype == torch.complex64

    mm_217 = mm_217.contiguous()
    view_788 = view_788.contiguous()
    freq_f32 = torch.view_as_real(view_788)

    out = torch.empty((1, 8192, 32, 128), dtype=torch.bfloat16, device=mm_217.device)

    N_ROWS = 8192
    N_HEADS = 32
    HEAD_DIM = 128
    BLOCK_HEADS = 32

    grid = (N_ROWS, triton.cdiv(N_HEADS, BLOCK_HEADS))
    _rope_kernel[grid](
        mm_217, freq_f32, out,
        N_HEADS, HEAD_DIM, BLOCK_HEADS,
        num_warps=4,
        num_stages=3,
    )
    return out