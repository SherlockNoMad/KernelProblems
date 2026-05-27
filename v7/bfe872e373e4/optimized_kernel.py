import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_kernel(
    in_ptr,
    cos_sin_ptr,
    out_ptr,
    BLOCK_S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D2: tl.constexpr,
):
    pid_s = tl.program_id(0)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    hd_offs = tl.arange(0, H * D)  # [H*D] = 4096

    # Each row s has H*D = 4096 contiguous bf16 elements
    in_base = s_offs[:, None] * (H * D) + hd_offs[None, :]
    x = tl.load(in_ptr + in_base).to(tl.float32)
    # Reshape: [BLOCK_S, H*D] -> [BLOCK_S, H, D2, 2]
    x = tl.reshape(x, (BLOCK_S, H, D2, 2))

    # Split real/imag via reshape -> [BLOCK_S, H, D2, 2], then slice
    x2 = tl.reshape(x, (BLOCK_S, H * D2, 2))
    x_real = tl.reshape(tl.sum(x2 * tl.reshape((tl.arange(0, 2) == 0).to(tl.float32), (1, 1, 2)), axis=2), (BLOCK_S, H, D2))
    x_imag = tl.reshape(tl.sum(x2 * tl.reshape((tl.arange(0, 2) == 1).to(tl.float32), (1, 1, 2)), axis=2), (BLOCK_S, H, D2))

    # Load cos_sin: shape [S, D2, 2] -> contiguous D = 128 per row
    cs_offs = tl.arange(0, D)  # 128 elements
    cs_base = s_offs[:, None] * D + cs_offs[None, :]
    cs = tl.load(cos_sin_ptr + cs_base)  # [BLOCK_S, D]
    cs = tl.reshape(cs, (BLOCK_S, D2, 2))
    c_real = tl.reshape(tl.sum(cs * tl.reshape((tl.arange(0, 2) == 0).to(tl.float32), (1, 1, 2)), axis=2), (BLOCK_S, D2))
    c_imag = tl.reshape(tl.sum(cs * tl.reshape((tl.arange(0, 2) == 1).to(tl.float32), (1, 1, 2)), axis=2), (BLOCK_S, D2))

    # Broadcast cos_sin across H
    c_real_b = c_real[:, None, :]
    c_imag_b = c_imag[:, None, :]

    out_real = x_real * c_real_b - x_imag * c_imag_b
    out_imag = x_real * c_imag_b + x_imag * c_real_b

    # Interleave back: [BLOCK_S, H, D2, 2] -> [BLOCK_S, H*D]
    out_combined = tl.join(out_real, out_imag)  # [BLOCK_S, H, D2, 2]
    out_flat = tl.reshape(out_combined, (BLOCK_S, H * D))

    tl.store(out_ptr + in_base, out_flat.to(out_ptr.dtype.element_ty))


def kernel_function(getitem_641, clone_63):
    assert getitem_641.dtype == torch.bfloat16
    assert clone_63.dtype == torch.complex64

    S = 8192
    H = 32
    D = 128
    D2 = 64

    cos_sin_f32 = torch.view_as_real(clone_63).contiguous()
    out = torch.empty((8192, 4096), dtype=torch.bfloat16, device=getitem_641.device)

    BLOCK_S = 4
    grid = (S // BLOCK_S,)
    _fused_rope_kernel[grid](
        getitem_641, cos_sin_f32, out,
        BLOCK_S=BLOCK_S, H=H, D=D, D2=D2,
        num_warps=8,
        num_stages=3,
    )
    return out