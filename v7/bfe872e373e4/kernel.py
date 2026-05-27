import torch
import triton
import triton.language as tl

# Fused kernel:
# Stages fused in one pass:
#  1. Read getitem_641 with logical layout [1,32,8192,128] and strides [33554432,128,4096,1]
#     and transpose dims 1,2 -> logical [1, 8192, 32, 128]
#  2. Cast bf16 -> fp32
#  3. View as complex pairs [1, 8192, 32, 64, 2]
#  4. Complex multiply with clone_63 (broadcast over dim2=32) of shape [1, 8192, 1, 64] complex64
#     stored as 2 floats (real, imag)
#  5. Cast back to bf16
#  6. Output viewed as [8192, 4096] (= [seq, 32*128])

@triton.jit
def _fused_rope_kernel(
    in_ptr,           # bf16, base of underlying storage
    cos_sin_ptr,      # complex64 reinterpreted as float32 pairs [1,8192,1,64,2]
    out_ptr,          # bf16, shape [8192, 4096]
    S,                # 8192
    H,                # 32
    D2,               # 64  (half of head_dim)
    BLOCK: tl.constexpr,
):
    # program over (seq_id, head_id)
    pid_s = tl.program_id(0)  # 0..S-1
    pid_h = tl.program_id(1)  # 0..H-1

    offs = tl.arange(0, BLOCK)  # we'll use first D2 lanes
    mask = offs < D2

    # input strides: [33554432, 128, 4096, 1] for [1,32,8192,128]
    # After transpose(1,2): logical [1,8192,32,128] with strides [33554432, 4096, 128, 1]
    # so input index for (s, h, d) = s*4096 + h*128 + d
    in_base = pid_s * 4096 + pid_h * 128

    # Load real and imag parts: pairs (d*2, d*2+1)
    real_offs = in_base + offs * 2
    imag_offs = in_base + offs * 2 + 1

    x_real = tl.load(in_ptr + real_offs, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(in_ptr + imag_offs, mask=mask, other=0.0).to(tl.float32)

    # cos_sin layout: [1, 8192, 1, 64] complex -> as float32 pairs [s, d, 2]
    # index for (s, d, 0/1) = s*64*2 + d*2 + (0/1)
    cs_base = pid_s * 64 * 2
    c_real = tl.load(cos_sin_ptr + cs_base + offs * 2, mask=mask, other=0.0)
    c_imag = tl.load(cos_sin_ptr + cs_base + offs * 2 + 1, mask=mask, other=0.0)

    # Complex multiply: (a+bi)*(c+di) = (ac - bd) + (ad + bc)i
    out_real = x_real * c_real - x_imag * c_imag
    out_imag = x_real * c_imag + x_imag * c_real

    # Output shape [8192, 4096], row = seq, col = head*128 + d_full
    # within head, real at 2*d, imag at 2*d+1
    out_base = pid_s * 4096 + pid_h * 128
    tl.store(out_ptr + out_base + offs * 2, out_real.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + out_base + offs * 2 + 1, out_imag.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(getitem_641, clone_63):
    """
    Fused RoPE-style complex multiply.
    Fuses: transpose + cast(fp32) + view_as_complex + complex mul + view_as_real
           + cast(bf16) + reshape to [8192, 4096], all in a single Triton kernel.
    """
    assert getitem_641.dtype == torch.bfloat16
    assert clone_63.dtype == torch.complex64
    assert getitem_641.is_cuda and clone_63.is_cuda

    # Underlying storage for getitem_641 is a 1D bf16 tensor of size 33554432
    # which is exactly what we got. Use it directly via the storage.
    # getitem_641.as_strided result: data_ptr points into base; we use it as-is.

    S = 8192
    H = 32
    D2 = 64  # head_dim / 2

    # Reinterpret clone_63 (complex64) as float32 view with last dim *2
    cos_sin_f32 = torch.view_as_real(clone_63).contiguous()  # shape [1,8192,1,64,2] float32
    # This is allocation/reinterpret only, not compute.

    out = torch.empty((8192, 4096), dtype=torch.bfloat16, device=getitem_641.device)

    BLOCK = 64
    grid = (S, H)
    _fused_rope_kernel[grid](
        getitem_641, cos_sin_f32, out,
        S, H, D2,
        BLOCK=BLOCK,
    )
    return out