import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    conj_ptr,
    input_ptr,
    out_ptr,
    BLOCK_S: tl.constexpr,
    H: tl.constexpr,
):
    pid_s = tl.program_id(0)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    d64 = tl.arange(0, 64)

    # conj layout (contiguous): [1, 8192, 1, 64, 2]
    # offset for (0, s, 0, d, 0/1) = s*128 + d*2 + (0/1)
    conj_real_offs = s_offs[:, None] * 128 + d64[None, :] * 2
    conj_imag_offs = conj_real_offs + 1
    c_real = tl.load(conj_ptr + conj_real_offs)
    c_imag = tl.load(conj_ptr + conj_imag_offs)

    for h in tl.static_range(H):
        # Input access uses original storage with strides [8388608, 128, 1024, 1]
        # for original shape [1, 8, 8192, 128].
        # After transpose(1,2) we want logical (0, s, h, d_full) which maps to
        # original (0, h, s, d_full): offset = h*128 + s*1024 + d_full
        # d_full = d*2 or d*2+1 (real/imag)
        base = s_offs[:, None] * 1024 + h * 128
        in_real_offs = base + d64[None, :] * 2
        in_imag_offs = in_real_offs + 1
        a_real = tl.load(input_ptr + in_real_offs).to(tl.float32)
        a_imag = tl.load(input_ptr + in_imag_offs).to(tl.float32)

        out_real = a_real * c_real - a_imag * c_imag
        out_imag = a_real * c_imag + a_imag * c_real

        # Output layout: [8192, 1024] contiguous
        # view_default_1 shape [1, 8192, 8, 128] -> view [1, 8192, 1024] -> [8192, 1024]
        # For (s, h, d_full): offset = s*1024 + h*128 + d_full
        out_base = s_offs[:, None] * 1024 + h * 128
        out_real_offs = out_base + d64[None, :] * 2
        out_imag_offs = out_real_offs + 1
        tl.store(out_ptr + out_real_offs, out_real.to(tl.bfloat16))
        tl.store(out_ptr + out_imag_offs, out_imag.to(tl.bfloat16))


def kernel_function(_conj_62, getitem_642):
    M = 8192
    H = 8

    output = torch.empty((M, 1024), dtype=torch.bfloat16, device=getitem_642.device)
    conj_real = torch.view_as_real(_conj_62).contiguous()

    BLOCK_S = 32
    grid = (M // BLOCK_S,)
    _fused_kernel[grid](
        conj_real, getitem_642, output,
        BLOCK_S=BLOCK_S,
        H=H,
        num_warps=4,
        num_stages=2,
    )

    return output