import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    conj_ptr,      # complex64: [1, 8192, 1, 64] -> stored as 2*float32
    input_ptr,     # bf16, with strides [_, 128, 1024, 1] for [1,8,8192,128]
    out_ptr,       # bf16: [8192, 1024]
    M,             # 8192 (seq)
    H,             # 8 (heads)
    D2,            # 64 (complex dim)
    in_stride_h,   # 128
    in_stride_s,   # 1024
    BLOCK: tl.constexpr,
):
    """
    Fused kernel performing:
      - transpose getitem_642 (1,2) implicitly via strides
      - cast bf16 -> fp32
      - view last dim as complex pairs (real, imag)
      - complex multiply with _conj_62 (broadcast over heads)
      - view back as real
      - cast to bf16
      - reshape to [8192, 1024]
    
    Each program handles one (seq, head) pair = 64 complex elements = 128 reals.
    """
    pid = tl.program_id(0)
    # Total programs = M * H
    s = pid // H
    h = pid % H
    
    # Indices for the 64 complex pairs
    d = tl.arange(0, BLOCK)  # BLOCK = 64
    mask = d < D2
    
    # Load conj_62 [1, s, 0, d] -> complex64 stored as (real, imag) interleaved
    # conj shape [1, 8192, 1, 64] complex => 2*float32 per complex
    # offset in float32: s * 64 * 2 + d * 2
    conj_base = s * D2 * 2 + d * 2
    c_real = tl.load(conj_ptr + conj_base, mask=mask, other=0.0)
    c_imag = tl.load(conj_ptr + conj_base + 1, mask=mask, other=0.0)
    
    # Load getitem_642 [1, h, s, 2*d:2*d+2] bf16
    # strides: dim1(h)=128, dim2(s)=1024, dim3(last)=1
    in_base = h * in_stride_h + s * in_stride_s
    a_real_bf = tl.load(input_ptr + in_base + d * 2, mask=mask, other=0.0)
    a_imag_bf = tl.load(input_ptr + in_base + d * 2 + 1, mask=mask, other=0.0)
    
    # Cast to fp32
    a_real = a_real_bf.to(tl.float32)
    a_imag = a_imag_bf.to(tl.float32)
    
    # Complex multiply: (a_real + a_imag*i) * (c_real + c_imag*i)
    out_real = a_real * c_real - a_imag * c_imag
    out_imag = a_real * c_imag + a_imag * c_real
    
    # Output [1, 8192, 8, 128] -> reshaped to [8192, 1024]
    # output index: row = s, col = h*128 + 2*d (real), h*128 + 2*d + 1 (imag)
    out_base = s * 1024 + h * 128
    tl.store(out_ptr + out_base + d * 2, out_real.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + out_base + d * 2 + 1, out_imag.to(tl.bfloat16), mask=mask)


def kernel_function(_conj_62, getitem_642):
    """
    Fused implementation of:
      clone _conj_62
      transpose getitem_642(1,2) -> [1,8192,8,128]
      to fp32, view_as_complex
      mul by _conj_62 (broadcasts over heads)
      view_as_real, to bf16
      reshape to [8192, 1024]
    
    All math in Triton; PyTorch only for allocation.
    """
    assert _conj_62.dtype == torch.complex64
    assert getitem_642.dtype == torch.bfloat16
    assert _conj_62.shape == (1, 8192, 1, 64)
    assert getitem_642.shape == (1, 8, 8192, 128)
    
    M = 8192
    H = 8
    D2 = 64  # complex dim
    
    # Strides of getitem_642: [8388608, 128, 1024, 1]
    in_stride_h = getitem_642.stride(1)  # 128
    in_stride_s = getitem_642.stride(2)  # 1024
    
    output = torch.empty((M, 1024), dtype=torch.bfloat16, device=getitem_642.device)
    
    # View complex64 as float32 (2x last dim)
    conj_real = torch.view_as_real(_conj_62).contiguous()  # [1, 8192, 1, 64, 2] float32
    
    BLOCK = 64
    grid = (M * H,)
    _fused_kernel[grid](
        conj_real, getitem_642, output,
        M, H, D2,
        in_stride_h, in_stride_s,
        BLOCK=BLOCK,
    )
    
    return output