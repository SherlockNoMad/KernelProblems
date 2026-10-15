"""One-launch indexed Q/K RoPE with PyTorch's complex-multiply rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_pairs(packed, frequency_real, frequency_imag):
    real = (packed << 16).to(tl.float32, bitcast=True)
    imag = (packed & 0xFFFF0000).to(tl.float32, bitcast=True)
    # CUDA's eager complex multiplication rounds these products before the FMA.
    imaginary_product = imag * frequency_imag
    # Arithmetic subtraction from zero would lose the sign of a zero product.
    negative_product = (imaginary_product.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    output_real = tl.fma(real, frequency_real, negative_product)
    output_imag = tl.fma(imag, frequency_real, real * frequency_imag)
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        constraints="=r,f,f",
        args=[output_real, output_imag],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _indexed_rope(
    Q,
    K,
    Position,
    Frequency,
    QOut,
    KOut,
    POSITION_STRIDE: tl.constexpr,
    FREQUENCY_ROW_STRIDE: tl.constexpr,
    FREQUENCY_PAIR_STRIDE: tl.constexpr,
):
    block = tl.program_id(0)
    is_query = block < 4096
    token = tl.where(is_query, block, block - 4096) * 2 + tl.arange(0, 2)
    position = tl.load(Position + token * POSITION_STRIDE).to(tl.int32)
    position = tl.where(position < 0, position + 8192, position)
    pair = tl.arange(0, 64)
    frequency_offset = position[:, None] * (FREQUENCY_ROW_STRIDE // 2) + pair[None, :] * (FREQUENCY_PAIR_STRIDE // 2)
    packed_frequency = tl.load(Frequency.to(tl.pointer_type(tl.uint64)) + frequency_offset)
    frequency_real = packed_frequency.to(tl.uint32).to(tl.float32, bitcast=True)
    frequency_imag = (packed_frequency >> 32).to(tl.uint32).to(tl.float32, bitcast=True)

    # Q and K use separate allocations, matching the eager output alias graph.
    if is_query:
        query_offset = token[:, None, None] * 2048 + tl.arange(0, 32)[None, :, None] * 64 + pair[None, None, :]
        packed_query = tl.load(Q.to(tl.pointer_type(tl.uint32)) + query_offset)
        output_query = _rotate_pairs(packed_query, frequency_real[:, None, :], frequency_imag[:, None, :])
        tl.store(QOut.to(tl.pointer_type(tl.uint32)) + query_offset, output_query)
    else:
        key_offset = token[:, None, None] * 512 + tl.arange(0, 8)[None, :, None] * 64 + pair[None, None, :]
        packed_key = tl.load(K.to(tl.pointer_type(tl.uint32)) + key_offset)
        output_key = _rotate_pairs(packed_key, frequency_real[:, None, :], frequency_imag[:, None, :])
        tl.store(KOut.to(tl.pointer_type(tl.uint32)) + key_offset, output_key)


def kernel_function(arg586_1, arg582_1, mm, mm_1):
    query = torch.empty((1, 8192, 32, 128), dtype=mm.dtype, device=mm.device)
    key = torch.empty((1, 8192, 8, 128), dtype=mm_1.dtype, device=mm_1.device)
    frequency = torch.view_as_real(arg582_1)
    _indexed_rope[(8192,)](
        mm, mm_1, arg586_1, frequency, query, key,
        arg586_1.stride(1), frequency.stride(0), frequency.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (query, key)


def roofline_workload(arg586_1, arg582_1, mm, mm_1):
    input_bytes = mm.numel() * mm.element_size() + mm_1.numel() * mm_1.element_size()
    frequency_bytes = torch.unique(arg586_1.remainder(8192)).numel() * 64 * 8
    read_bytes = input_bytes + arg586_1.numel() * arg586_1.element_size() + frequency_bytes
    return {
        "read_bytes": read_bytes,
        "write_bytes": input_bytes,
        "flops": 6 * (mm.numel() + mm_1.numel()) // 2,
        "special_ops": 0,
        "working_set_bytes": read_bytes + input_bytes,
        "notes": "Each BF16 Q/K value is read and written once; positions and only selected complex64 frequency rows are counted. Two FP32 products and two FMAs per complex pair.",
    }
