"""Fused RoPE with the FP32 operation order used by eager CUDA complex mul."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_pairs(
    input_pairs,
    frequency_pairs,
    output_pairs,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
):
    sequence = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    pair = tl.arange(0, HEADS * 64)
    offsets = sequence[:, None] * (HEADS * 64) + pair[None, :]

    packed_input = tl.load(input_pairs + offsets).to(tl.uint32)
    input_real = (packed_input & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    input_imag = (packed_input >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)

    frequency_offsets = sequence[:, None] * 64 + tl.arange(0, 64)[None, :]
    packed_frequency = tl.load(frequency_pairs + frequency_offsets).to(tl.uint64)
    frequency_real = (packed_frequency & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    frequency_imag = (packed_frequency >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    frequency_real = tl.reshape(
        tl.broadcast_to(frequency_real[:, None, :], (ROWS, HEADS, 64)),
        (ROWS, HEADS * 64),
    )
    frequency_imag = tl.reshape(
        tl.broadcast_to(frequency_imag[:, None, :], (ROWS, HEADS, 64)),
        (ROWS, HEADS * 64),
    )

    # Eager CUDA rounds these products before the two fused multiply-adds.
    imag_product = input_imag * frequency_imag
    # Flip the sign bit: arithmetic 0 - product would change a positive zero.
    neg_imag_product = (imag_product.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    output_real = tl.fma(input_real, frequency_real, neg_imag_product)
    output_imag = tl.fma(input_imag, frequency_real, input_real * frequency_imag)
    real_bits = output_real.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    imag_bits = output_imag.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    tl.store(output_pairs + offsets, real_bits | (imag_bits << 16))


def kernel_function(mm_217, view_788):
    input_pairs = mm_217.view(torch.int32)
    output = torch.empty((1, 8192, 32, 128), dtype=torch.bfloat16, device=mm_217.device)
    _rope_pairs[(4096,)](
        input_pairs, view_788.view(torch.int64), output.view(torch.int32),
        HEADS=32, ROWS=2, num_warps=4, enable_fp_fusion=False,
    )
    return output


def roofline_workload(*inputs):
    input_bytes = 8192 * 32 * 128 * 2
    frequency_bytes = 8192 * 64 * 8
    return {
        "read_bytes": input_bytes + frequency_bytes,
        "write_bytes": input_bytes,
        "flops": 6 * 8192 * 32 * 64,
        "special_ops": 0,
        "working_set_bytes": 2 * input_bytes + frequency_bytes,
        "notes": "One BF16 input and output, one shared complex64 frequency per sequence/pair; four FP32 multiplies and two adds per complex pair.",
    }
