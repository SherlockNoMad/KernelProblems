import torch
import triton
import triton.language as tl

# Fused kernel:
# - Reshape mm_217 [8192, 4096] -> [1, 8192, 32, 64, 2] (last dim = complex pair)
# - Cast to f32, treat as complex
# - Multiply by view_788 (broadcast over heads dim 32): shape [1, 8192, 1, 64] complex
# - Cast back to bf16, output shape [1, 8192, 32, 128]
#
# Layout: input mm_217 is contiguous [8192, 4096]. After view [1,8192,32,128]
# linear index: row=8192_idx, then within row of 4096 elements:
#   head h (0..31), then dim d (0..127). Position in row = h*128 + d
# For complex pair index p (0..63): real at d=2p, imag at d=2p+1
# view_788 is [1, 8192, 1, 64] complex64, contiguous. Layout: row*64 + p (complex elements)
# Output [1, 8192, 32, 128] bf16, contiguous: same layout as input.

@triton.jit
def _rope_kernel(
    inp_ptr,        # bf16, [8192, 4096]
    freq_ptr,       # complex64 viewed as float32 pairs, [1, 8192, 1, 64, 2]
    out_ptr,        # bf16, [1, 8192, 32, 128]
    N_ROWS: tl.constexpr,      # 8192
    N_HEADS: tl.constexpr,     # 32
    N_PAIRS: tl.constexpr,     # 64
    BLOCK_PAIRS: tl.constexpr, # block size over pairs
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    pid_p = tl.program_id(2)

    p_offs = pid_p * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    mask = p_offs < N_PAIRS

    # Input offsets: row * 4096 + head * 128 + (2*p), (2*p+1)
    row_base = row * (N_HEADS * N_PAIRS * 2) + head * (N_PAIRS * 2)
    real_off = row_base + 2 * p_offs
    imag_off = row_base + 2 * p_offs + 1

    x_real = tl.load(inp_ptr + real_off, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(inp_ptr + imag_off, mask=mask, other=0.0).to(tl.float32)

    # freq offsets: shape [1, 8192, 1, 64] complex -> as float32 [.., 64, 2]
    # freq is broadcast over heads
    freq_base = row * (N_PAIRS * 2)
    f_real_off = freq_base + 2 * p_offs
    f_imag_off = freq_base + 2 * p_offs + 1
    f_real = tl.load(freq_ptr + f_real_off, mask=mask, other=0.0)
    f_imag = tl.load(freq_ptr + f_imag_off, mask=mask, other=0.0)

    # complex multiply: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    # Store to output: same layout as input
    out_base = row * (N_HEADS * N_PAIRS * 2) + head * (N_PAIRS * 2)
    o_real_off = out_base + 2 * p_offs
    o_imag_off = out_base + 2 * p_offs + 1

    tl.store(out_ptr + o_real_off, out_real.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + o_imag_off, out_imag.to(tl.bfloat16), mask=mask)


def kernel_function(mm_217, view_788):
    """
    Fused RoPE-style complex multiplication:
      - View mm_217 [8192,4096] as [1,8192,32,64] complex (via float32 pair view)
      - Multiply by view_788 [1,8192,1,64] complex (broadcasting over heads)
      - Cast back to bf16, output [1,8192,32,128]
    Entire pipeline fused into a single Triton kernel.
    """
    assert mm_217.is_cuda and view_788.is_cuda
    assert mm_217.dtype == torch.bfloat16
    assert view_788.dtype == torch.complex64
    assert mm_217.shape == (8192, 4096)
    assert view_788.shape == (1, 8192, 1, 64)

    mm_217 = mm_217.contiguous()
    view_788 = view_788.contiguous()

    # View complex64 as float32 with last dim 2
    freq_f32 = torch.view_as_real(view_788)  # [1,8192,1,64,2] float32, contiguous view

    out = torch.empty((1, 8192, 32, 128), dtype=torch.bfloat16, device=mm_217.device)

    N_ROWS = 8192
    N_HEADS = 32
    N_PAIRS = 64
    BLOCK_PAIRS = 64

    grid = (N_ROWS, N_HEADS, triton.cdiv(N_PAIRS, BLOCK_PAIRS))
    _rope_kernel[grid](
        mm_217, freq_f32, out,
        N_ROWS, N_HEADS, N_PAIRS, BLOCK_PAIRS,
    )
    return out