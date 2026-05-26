import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fused_kernel(
    input_ptr,
    conj_ptr,
    output_ptr,
    S, H,
    BLOCK_S: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_s = tl.program_id(1)

    h = pid_d // 64
    j_pair = pid_d % 64

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    input_base = h * 128 + s_offs * (H * 128) + j_pair * 2
    a = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)

    conj_base = s_offs * 128 + j_pair * 2
    c = tl.load(conj_ptr + conj_base, mask=mask, other=0.0)
    d = tl.load(conj_ptr + conj_base + 1, mask=mask, other=0.0)

    out_real = a * c - b * d
    out_imag = a * d + b * c

    d_real = h * 128 + j_pair * 2
    d_imag = d_real + 1

    out_real_offs = d_real * S + s_offs
    out_imag_offs = d_imag * S + s_offs

    tl.store(output_ptr + out_real_offs, out_real.to(tl.bfloat16), mask=mask)
    tl.store(output_ptr + out_imag_offs, out_imag.to(tl.bfloat16), mask=mask)


@triton.jit
def _transpose_smem_kernel(
    input_ptr,
    output_ptr,
    S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    D = H * 128
    mask_s = s_offs < S
    mask_d = d_offs < D

    h_idx = d_offs // 128
    dd_idx = d_offs % 128
    in_offs = h_idx[None, :] * 128 + s_offs[:, None] * D + dd_idx[None, :]
    mask_in = mask_s[:, None] & mask_d[None, :]
    x = tl.load(input_ptr + in_offs, mask=mask_in, other=0.0)

    x_t = tl.trans(x)

    out_offs = d_offs[:, None] * S + s_offs[None, :]
    mask_out = mask_d[:, None] & mask_s[None, :]
    tl.store(output_ptr + out_offs, x_t, mask=mask_out)


def kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643):
    assert torch.cuda.is_available()
    S = 8192

    t0 = torch.empty((4096, S), device='cuda', dtype=torch.bfloat16)
    t1 = torch.empty((1024, S), device='cuda', dtype=torch.bfloat16)
    t2 = torch.empty((1024, S), device='cuda', dtype=torch.bfloat16)

    conj63_f32 = torch.view_as_real(_conj_63).contiguous()
    conj62_f32 = torch.view_as_real(_conj_62).contiguous()

    BLOCK_S = 256

    grid1 = (32 * 64, triton.cdiv(S, BLOCK_S))
    _rope_fused_kernel[grid1](
        getitem_641, conj63_f32, t0, S, 32, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=3,
    )

    grid2 = (8 * 64, triton.cdiv(S, BLOCK_S))
    _rope_fused_kernel[grid2](
        getitem_642, conj62_f32, t1, S, 8, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=3,
    )

    BLOCK_S3 = 64
    BLOCK_D3 = 64
    D3 = 1024
    grid3 = (triton.cdiv(S, BLOCK_S3), triton.cdiv(D3, BLOCK_D3))
    _transpose_smem_kernel[grid3](
        getitem_643, t2, S, 8, BLOCK_S=BLOCK_S3, BLOCK_D=BLOCK_D3,
        num_warps=4, num_stages=2,
    )

    return (t0, t1, t2)