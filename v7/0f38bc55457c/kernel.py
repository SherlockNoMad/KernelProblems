import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    mm_ptr,           # [8192, 4096] bf16
    mm1_ptr,          # [8192, 1024] bf16
    idx_ptr,          # [1, 8192] int32
    freq_ptr,         # [8192, 64] complex64 (treated as [8192, 128] float32)
    out1_ptr,         # [1, 8192, 32, 128] float32
    out2_ptr,         # [1, 8192, 8, 128] bf16
    S,                # seq_len = 8192
    H1: tl.constexpr, # 32
    H2: tl.constexpr, # 8
    D: tl.constexpr,  # 128
    HD: tl.constexpr, # 64 (D//2)
    BLOCK_S: tl.constexpr,
):
    """
    Fused RoPE: 
      - Load mm reshaped to [S, 32, 128], mm1 to [S, 8, 128]
      - Treat last dim as 64 complex pairs
      - Load freq[idx[s]] as 64 complex pairs
      - Multiply complex * complex
      - Store as float32 (out1) and bf16 (out2)
    """
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)  # 0..(H1+H2)-1, first H1 for out1, next H2 for out2
    
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    
    # Load index for these s positions
    idx_vals = tl.load(idx_ptr + s_offs, mask=s_mask, other=0).to(tl.int32)
    
    # Load freq pairs (real, imag) interleaved. freq is [8192, 64] complex64
    # = [8192, 128] float32 with stride (128, 1)
    d_offs = tl.arange(0, HD)  # 0..63
    # Real part at position 2*d, imag at 2*d+1
    freq_base = idx_vals[:, None] * (D) + d_offs[None, :] * 2
    freq_re = tl.load(freq_ptr + freq_base, mask=s_mask[:, None], other=0.0)
    freq_im = tl.load(freq_ptr + freq_base + 1, mask=s_mask[:, None], other=0.0)
    
    if pid_h < H1:
        h = pid_h
        # mm[s, h*128 + d] -> reshape: input has stride [4096, 1] for [8192, 4096]
        # Element [s, h, d] = mm[s, h*128 + d]
        base = s_offs[:, None] * 4096 + h * D + d_offs[None, :] * 2
        x_re_bf = tl.load(mm_ptr + base, mask=s_mask[:, None], other=0.0)
        x_im_bf = tl.load(mm_ptr + base + 1, mask=s_mask[:, None], other=0.0)
        x_re = x_re_bf.to(tl.float32)
        x_im = x_im_bf.to(tl.float32)
        
        # Complex multiply: (x_re + i*x_im) * (f_re + i*f_im)
        # = (x_re*f_re - x_im*f_im) + i*(x_re*f_im + x_im*f_re)
        out_re = x_re * freq_re - x_im * freq_im
        out_im = x_re * freq_im + x_im * freq_re
        
        # Store to out1: [1, 8192, 32, 128] float32
        # Element [0, s, h, 2d] = out_re, [0, s, h, 2d+1] = out_im
        out_base = s_offs[:, None] * (H1 * D) + h * D + d_offs[None, :] * 2
        tl.store(out1_ptr + out_base, out_re, mask=s_mask[:, None])
        tl.store(out1_ptr + out_base + 1, out_im, mask=s_mask[:, None])
    else:
        h = pid_h - H1
        # mm1[s, h*128 + d]
        base = s_offs[:, None] * 1024 + h * D + d_offs[None, :] * 2
        x_re_bf = tl.load(mm1_ptr + base, mask=s_mask[:, None], other=0.0)
        x_im_bf = tl.load(mm1_ptr + base + 1, mask=s_mask[:, None], other=0.0)
        x_re = x_re_bf.to(tl.float32)
        x_im = x_im_bf.to(tl.float32)
        
        out_re = x_re * freq_re - x_im * freq_im
        out_im = x_re * freq_im + x_im * freq_re
        
        # Store to out2: [1, 8192, 8, 128] bf16
        out_base = s_offs[:, None] * (H2 * D) + h * D + d_offs[None, :] * 2
        tl.store(out2_ptr + out_base, out_re.to(tl.bfloat16), mask=s_mask[:, None])
        tl.store(out2_ptr + out_base + 1, out_im.to(tl.bfloat16), mask=s_mask[:, None])


def kernel_function(mm, mm_1, arg586_1, arg582_1):
    """
    Fused RoPE-like operation:
      Stage 1: View mm as [1, S, 32, 128] and mm_1 as [1, S, 8, 128]
      Stage 2: Cast to float32 and interpret as complex pairs
      Stage 3: Gather freq table via arg586_1 indices
      Stage 4: Complex multiply with broadcast over heads
      Stage 5: Store real view -> out1 (float32), out2 (bf16)
    All fused into a single Triton kernel.
    """
    assert mm.is_cuda and mm_1.is_cuda
    assert mm.dtype == torch.bfloat16
    assert mm_1.dtype == torch.bfloat16
    assert arg582_1.dtype == torch.complex64
    
    S = 8192
    H1 = 32
    H2 = 8
    D = 128
    HD = 64
    
    # Ensure contiguous
    mm_c = mm.contiguous()
    mm1_c = mm_1.contiguous()
    idx_c = arg586_1.contiguous()
    # View complex tensor as float32 with last dim doubled
    freq_real = torch.view_as_real(arg582_1.contiguous()).contiguous()  # [8192, 64, 2] float32
    
    out1 = torch.empty((1, S, H1, D), dtype=torch.float32, device=mm.device)
    out2 = torch.empty((1, S, H2, D), dtype=torch.bfloat16, device=mm.device)
    
    BLOCK_S = 16
    grid = (triton.cdiv(S, BLOCK_S), H1 + H2)
    
    _rope_kernel[grid](
        mm_c, mm1_c, idx_c, freq_real,
        out1, out2,
        S, H1, H2, D, HD,
        BLOCK_S=BLOCK_S,
    )
    
    return (out1, out2)