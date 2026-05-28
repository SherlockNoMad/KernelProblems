import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    idx_ptr,
    freqs_ptr,
    in_ptr,
    out_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    idx = tl.load(idx_ptr + s_offs).to(tl.int64)  # [BLOCK_S]

    F2 = F * 2
    offs_2f = tl.arange(0, F * 2)

    # Load freqs: [BLOCK_S, 2F]
    freq_offsets = idx[:, None] * F2 + offs_2f[None, :]
    freqs = tl.load(freqs_ptr + freq_offsets)
    freqs_r = tl.reshape(freqs, (BLOCK_S, F, 2))
    c, d = tl.split(freqs_r)  # each [BLOCK_S, F]

    for h_local in tl.static_range(0, BLOCK_H):
        h = pid_h * BLOCK_H + h_local
        if h < H:
            base = s_offs[:, None] * (H * F2) + h * F2 + offs_2f[None, :]
            x = tl.load(in_ptr + base).to(tl.float32)
            x_r = tl.reshape(x, (BLOCK_S, F, 2))
            a, b = tl.split(x_r)

            real_out = a * c - b * d
            imag_out = a * d + b * c

            out_pair = tl.join(real_out, imag_out)
            out_flat = tl.reshape(out_pair, (BLOCK_S, F * 2))
            tl.store(out_ptr + base, out_flat.to(tl.bfloat16))


def kernel_function(arg586_1, arg582_1, mm, mm_1):
    device = arg586_1.device
    S = 8192
    F = 64

    out_q = torch.empty((1, S, 32, 128), dtype=torch.bfloat16, device=device)
    out_k = torch.empty((1, S, 8, 128), dtype=torch.bfloat16, device=device)

    freqs_f32 = torch.view_as_real(arg582_1).contiguous()

    idx_flat = arg586_1.view(-1).contiguous()
    mm_c = mm.contiguous()
    mm1_c = mm_1.contiguous()

    # Q: H=32
    BLOCK_S_Q = 32
    BLOCK_H_Q = 8
    grid_q = (S // BLOCK_S_Q, (32 + BLOCK_H_Q - 1) // BLOCK_H_Q)
    _rope_kernel[grid_q](
        idx_flat, freqs_f32, mm_c, out_q,
        H=32, F=F, BLOCK_H=BLOCK_H_Q, BLOCK_S=BLOCK_S_Q,
        num_warps=8, num_stages=3,
    )

    # K: H=8
    BLOCK_S_K = 32
    BLOCK_H_K = 8
    grid_k = (S // BLOCK_S_K, 1)
    _rope_kernel[grid_k](
        idx_flat, freqs_f32, mm1_c, out_k,
        H=8, F=F, BLOCK_H=BLOCK_H_K, BLOCK_S=BLOCK_S_K,
        num_warps=8, num_stages=3,
    )

    return (out_q, out_k)