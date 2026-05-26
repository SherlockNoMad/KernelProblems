import torch
import triton
import triton.language as tl


@triton.jit
def _rope_transpose_kernel(
    input_ptr,    # bf16, shape [S, H, 128] contiguous in last 2 dims
    conj_ptr,     # float32 view of complex64, shape [S, 64, 2]
    output_ptr,   # bf16, shape [H*128, S]
    S, H,
    BLOCK_S: tl.constexpr,
):
    pid_d = tl.program_id(0)  # 0..H*64-1
    pid_s = tl.program_id(1)

    h = pid_d // 64
    j_pair = pid_d % 64

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    # Input layout: after transpose(1,2), shape [1, S, H, 128].
    # contiguous-style indexing: idx = s * (H*128) + h*128 + 2*j_pair (+0/+1)
    input_base = s_offs * (H * 128) + h * 128 + j_pair * 2
    a = tl.load(input_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)

    # conj layout: [S, 64, 2] -> idx = s*128 + j_pair*2 (+0/+1)
    conj_base = s_offs * 128 + j_pair * 2
    c = tl.load(conj_ptr + conj_base, mask=mask, other=0.0)
    d = tl.load(conj_ptr + conj_base + 1, mask=mask, other=0.0)

    # Complex multiply (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
    out_real = a * c - b * d
    out_imag = a * d + b * c

    # Output [H*128, S]: store at row (h*128+2*j_pair), col s
    d_real = h * 128 + j_pair * 2
    d_imag = d_real + 1

    out_real_offs = d_real * S + s_offs
    out_imag_offs = d_imag * S + s_offs

    tl.store(output_ptr + out_real_offs, out_real.to(tl.bfloat16), mask=mask)
    tl.store(output_ptr + out_imag_offs, out_imag.to(tl.bfloat16), mask=mask)


@triton.jit
def _transpose_kernel(
    input_ptr,   # bf16, shape [S, H, 128] (already transposed view)
    output_ptr,  # bf16, shape [H*128, S]
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
    mask = mask_s[:, None] & mask_d[None, :]

    in_offs = s_offs[:, None] * D + d_offs[None, :]
    x = tl.load(input_ptr + in_offs, mask=mask, other=0.0)

    out_offs = d_offs[None, :] * S + s_offs[:, None]
    tl.store(output_ptr + out_offs, x, mask=mask)


def kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643):
    assert torch.cuda.is_available()
    S = 8192

    # Outputs
    t0 = torch.empty((4096, S), device='cuda', dtype=torch.bfloat16)
    t1 = torch.empty((1024, S), device='cuda', dtype=torch.bfloat16)
    t2 = torch.empty((1024, S), device='cuda', dtype=torch.bfloat16)

    # The transposed view (1,2): [1, S, H, 128] - need contiguous version
    # getitem_641: [1, 32, 8192, 128] -> transpose -> [1, 8192, 32, 128]
    # We need the float32 cast of these, but we can do the bf16->fp32 conversion in kernel.
    # However the input strides may not be contiguous on the [S,H,128] view.
    # Make contiguous copies of the transposed views.
    inp1 = getitem_641.transpose(1, 2).contiguous()  # [1, S, 32, 128]
    inp2 = getitem_642.transpose(1, 2).contiguous()  # [1, S, 8, 128]
    inp3 = getitem_643.transpose(1, 2).contiguous()  # [1, S, 8, 128]

    # _conj_63: [1, 8192, 1, 64] complex64 -> view as float32 [1, 8192, 1, 64, 2] -> [S, 64, 2]
    conj63_f32 = torch.view_as_real(_conj_63).contiguous()  # [1, 8192, 1, 64, 2]
    conj62_f32 = torch.view_as_real(_conj_62).contiguous()

    BLOCK_S = 64

    # Kernel 1: H=32
    grid1 = (32 * 64, triton.cdiv(S, BLOCK_S))
    _rope_transpose_kernel[grid1](
        inp1, conj63_f32, t0, S, 32, BLOCK_S=BLOCK_S,
    )

    # Kernel 2: H=8
    grid2 = (8 * 64, triton.cdiv(S, BLOCK_S))
    _rope_transpose_kernel[grid2](
        inp2, conj62_f32, t1, S, 8, BLOCK_S=BLOCK_S,
    )

    # Kernel 3: just transpose
    BLOCK_S3 = 32
    BLOCK_D3 = 64
    D3 = 1024
    grid3 = (triton.cdiv(S, BLOCK_S3), triton.cdiv(D3, BLOCK_D3))
    _transpose_kernel[grid3](
        inp3, t2, S, 8, BLOCK_S=BLOCK_S3, BLOCK_D=BLOCK_D3,
    )

    return (t0, t1, t2)