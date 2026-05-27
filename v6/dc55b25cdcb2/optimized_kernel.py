import torch
import triton
import triton.language as tl


@triton.jit
def _fused_complex_mul_kernel(
    in_ptr,
    conj_ptr,
    out_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (s, h) pair, processing all K complex elements (BLOCK_K = K)
    pid = tl.program_id(0)
    h = pid % H
    s = pid // H

    k = tl.arange(0, BLOCK_K)

    # Input layout: contiguous [S, H, 2*K]; load 2*K bf16 elements as real/imag interleaved
    base_in = s * (H * 2 * K) + h * (2 * K)
    ir = tl.load(in_ptr + base_in + 2 * k).to(tl.float32)
    ii = tl.load(in_ptr + base_in + 2 * k + 1).to(tl.float32)

    # Conj layout: [S, 2*K] float32
    base_c = s * (2 * K)
    cr = tl.load(conj_ptr + base_c + 2 * k)
    ci = tl.load(conj_ptr + base_c + 2 * k + 1)

    out_r = ir * cr - ii * ci
    out_i = ir * ci + ii * cr

    base_out = s * (H * 2 * K) + h * (2 * K)
    tl.store(out_ptr + base_out + 2 * k, out_r.to(tl.bfloat16))
    tl.store(out_ptr + base_out + 2 * k + 1, out_i.to(tl.bfloat16))


@triton.jit
def _fused_complex_mul_kernel_v2(
    in_ptr,
    conj_ptr,
    out_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
):
    # Each program handles HEADS_PER_BLOCK heads for one s
    pid = tl.program_id(0)
    num_h_blocks = H // HEADS_PER_BLOCK
    s = pid // num_h_blocks
    h_block = pid % num_h_blocks
    h_start = h_block * HEADS_PER_BLOCK

    # Load conj for this s: K complex elements = 2*K floats
    k_idx = tl.arange(0, K)
    base_c = s * (2 * K)
    cr = tl.load(conj_ptr + base_c + 2 * k_idx)  # [K]
    ci = tl.load(conj_ptr + base_c + 2 * k_idx + 1)  # [K]

    # Process HEADS_PER_BLOCK heads
    h_offs = h_start + tl.arange(0, HEADS_PER_BLOCK)  # [HPB]
    # Build 2D indices: [HPB, K]
    base_sh = s * (H * 2 * K)
    # Real part offsets: base_sh + h*(2K) + 2k
    h2d = h_offs[:, None] * (2 * K)  # [HPB, 1]
    k2d_r = 2 * k_idx[None, :]  # [1, K]
    k2d_i = 2 * k_idx[None, :] + 1

    ir = tl.load(in_ptr + base_sh + h2d + k2d_r).to(tl.float32)
    ii = tl.load(in_ptr + base_sh + h2d + k2d_i).to(tl.float32)

    cr_b = cr[None, :]
    ci_b = ci[None, :]

    out_r = ir * cr_b - ii * ci_b
    out_i = ir * ci_b + ii * cr_b

    tl.store(out_ptr + base_sh + h2d + k2d_r, out_r.to(tl.bfloat16))
    tl.store(out_ptr + base_sh + h2d + k2d_i, out_i.to(tl.bfloat16))


def kernel_function(_conj_63, getitem_641):
    assert _conj_63.is_cuda and getitem_641.is_cuda
    assert _conj_63.dtype == torch.complex64
    assert getitem_641.dtype == torch.bfloat16

    S = 8192
    H = 32
    K = 64
    D = 2 * K

    in_t = getitem_641.transpose(1, 2).contiguous()

    conj_contig = _conj_63.contiguous()
    conj_f32 = torch.view_as_real(conj_contig)

    out = torch.empty((S, H * D), dtype=torch.bfloat16, device=getitem_641.device)

    HEADS_PER_BLOCK = 8
    grid = (S * (H // HEADS_PER_BLOCK),)

    _fused_complex_mul_kernel_v2[grid](
        in_t,
        conj_f32,
        out,
        S=S,
        H=H,
        K=K,
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        num_warps=4,
    )

    return out