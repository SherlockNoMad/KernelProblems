import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_complex_mul_kernel(
    in_ptr, clone_ptr, out_ptr,
    S, H, D_HALF,  # S=8192, H=8, D_HALF=64
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: transpose + cast f32 + view as complex + complex mul + 
    view as real + cast bf16 + reshape.
    
    Each program handles one (s, h) pair, processing all 64 complex pairs.
    """
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k = tl.arange(0, BLOCK_K)
    mask = k < D_HALF
    
    # Input layout (after transpose, which matches storage): [1, 8192, 8, 128] contiguous
    # offset for (s, h, 2k) and (s, h, 2k+1)
    base_in = pid_s * (H * D_HALF * 2) + pid_h * (D_HALF * 2)
    a_r = tl.load(in_ptr + base_in + 2 * k, mask=mask, other=0.0).to(tl.float32)
    a_i = tl.load(in_ptr + base_in + 2 * k + 1, mask=mask, other=0.0).to(tl.float32)
    
    # clone_62 layout: [1, 8192, 1, 64] complex64 -> interleaved float32 [s*64+k]*2
    base_c = pid_s * D_HALF * 2
    b_r = tl.load(clone_ptr + base_c + 2 * k, mask=mask, other=0.0).to(tl.float32)
    b_i = tl.load(clone_ptr + base_c + 2 * k + 1, mask=mask, other=0.0).to(tl.float32)
    
    out_r = a_r * b_r - a_i * b_i
    out_i = a_r * b_i + a_i * b_r
    
    # Output: [8192, 1024] = [s, h*128 + d]; d = 2k or 2k+1
    base_out = pid_s * (H * D_HALF * 2) + pid_h * (D_HALF * 2)
    tl.store(out_ptr + base_out + 2 * k, out_r.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + base_out + 2 * k + 1, out_i.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(getitem_642, clone_62):
    """
    Fused implementation of: transpose -> cast f32 -> view as complex ->
    complex multiply with clone_62 -> view as real -> cast bf16 -> reshape.
    
    Returns tensor of shape [8192, 1024] bf16.
    """
    assert getitem_642.is_cuda and clone_62.is_cuda
    assert getitem_642.dtype == torch.bfloat16
    assert clone_62.dtype == torch.complex64
    
    # getitem_642 shape [1, 8, 8192, 128] with strides [8388608, 128, 1024, 1]
    # In memory, it's laid out as [1, 8192, 8, 128] contiguous (the storage).
    S = 8192
    H = 8
    D_HALF = 64
    
    # Use the underlying storage view as [S, H, 128]
    # The storage tensor is 1D of size 8388608. We can index it directly.
    in_flat = getitem_642  # we'll pass the base pointer; layout matches storage
    
    # clone_62 is contiguous complex64 of shape [1, 8192, 1, 64]
    clone_flat = clone_62.contiguous()
    # View complex64 as float32 (interleaved real, imag); just pass pointer
    clone_as_real = torch.view_as_real(clone_flat)  # [1, 8192, 1, 64, 2] f32, no compute
    
    output = torch.empty((S, H * D_HALF * 2), dtype=torch.bfloat16, device=getitem_642.device)
    
    BLOCK_K = 64
    grid = (S, H)
    _rotary_complex_mul_kernel[grid](
        in_flat, clone_as_real, output,
        S, H, D_HALF,
        BLOCK_K=BLOCK_K,
    )
    return output