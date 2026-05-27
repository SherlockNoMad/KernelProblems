import torch
import triton
import triton.language as tl


@triton.jit
def _fused_complex_mul_kernel(
    in_ptr,       # bfloat16, logical shape [S, H, D=128], contiguous after transpose
    conj_ptr,     # float32 view of complex64 [1, S, 1, K=64] -> 2*K floats per s
    out_ptr,      # bfloat16, shape [S, H*D] = [8192, 4096]
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,   # 64
    BLOCK: tl.constexpr,
):
    # Each program handles BLOCK complex elements (each = 2 bf16 / 2 fp32)
    pid = tl.program_id(0)
    total = S * H * K  # number of complex elements
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Decompose linear offset into (s, h, k)
    k = offs % K
    hs = offs // K
    h = hs % H
    s = hs // H

    # Load real/imag from input (which is bf16). Element layout: in[s, h, 2k], in[s, h, 2k+1]
    base_in = s * (H * 2 * K) + h * (2 * K) + 2 * k
    ir = tl.load(in_ptr + base_in, mask=mask, other=0.0).to(tl.float32)
    ii = tl.load(in_ptr + base_in + 1, mask=mask, other=0.0).to(tl.float32)

    # Load conj real/imag from float32 view: conj[s, k] -> offset s*(2K) + 2k
    base_c = s * (2 * K) + 2 * k
    cr = tl.load(conj_ptr + base_c, mask=mask, other=0.0)
    ci = tl.load(conj_ptr + base_c + 1, mask=mask, other=0.0)

    # Complex multiply: (ir + i*ii) * (cr + i*ci)
    out_r = ir * cr - ii * ci
    out_i = ir * ci + ii * cr

    # Store to output [s, h*D + 2k], [s, h*D + 2k+1] where D = 2K
    base_out = s * (H * 2 * K) + h * (2 * K) + 2 * k
    tl.store(out_ptr + base_out, out_r.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + base_out + 1, out_i.to(tl.bfloat16), mask=mask)


def kernel_function(_conj_63, getitem_641):
    """
    Fused kernel implementing:
      1. transpose getitem_641 dims (1,2) -> [1, 8192, 32, 128]
      2. cast to fp32
      3. view as complex [1, 8192, 32, 64]
      4. multiply by _conj_63 (broadcast over head dim)
      5. view back as real, cast to bf16
      6. reshape to [8192, 4096]
    All compute is done in a single Triton kernel.
    """
    assert _conj_63.is_cuda and getitem_641.is_cuda
    assert _conj_63.dtype == torch.complex64
    assert getitem_641.dtype == torch.bfloat16

    # Shapes
    # _conj_63: [1, 8192, 1, 64] complex64
    # getitem_641: [1, 32, 8192, 128] strided so that transpose(1,2) is contiguous
    S = 8192
    H = 32
    K = 64  # complex elements per head
    D = 2 * K  # 128

    # Make input contiguous in the transposed layout [1, 8192, 32, 128]
    # The provided as_strided already gives transposed-friendly memory:
    # strides [33554432, 128, 4096, 1] for shape [1,32,8192,128]
    # After transpose(1,2): shape [1,8192,32,128], strides [33554432, 4096, 128, 1] -> contiguous
    in_t = getitem_641.transpose(1, 2).contiguous()  # ensures contiguous bf16 layout

    # View conj tensor as float32 pairs. complex64 is stored as (real, imag) float32 pairs contiguously.
    # _conj_63 shape [1, 8192, 1, 64] -> underlying float32 length = 8192 * 64 * 2
    conj_contig = _conj_63.contiguous()
    conj_f32 = torch.view_as_real(conj_contig)  # [1, 8192, 1, 64, 2] float32, contiguous
    # This is just a view, no compute.

    # Allocate output
    out = torch.empty((S, H * D), dtype=torch.bfloat16, device=getitem_641.device)

    total_complex = S * H * K
    BLOCK = 1024
    grid = (triton.cdiv(total_complex, BLOCK),)

    _fused_complex_mul_kernel[grid](
        in_t,
        conj_f32,
        out,
        S=S,
        H=H,
        K=K,
        BLOCK=BLOCK,
        num_warps=4,
    )

    return out