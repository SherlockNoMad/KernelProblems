"""Bit-exact native-order RMSNorm and materialized packed-parameter copies.

The RMS reduction follows PyTorch's 128-thread, vec4 CUDA reduction for
4096 BF16 features. Each output parameter owns independent storage.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def _normalized_rows(input_tensor, residual, token_ids, packed_parameters,
                     normalized, PACKED_ROW_STRIDE: gl.constexpr,
                     EMBEDDING: gl.constexpr):
    row = gl.program_id(0)
    layout: gl.constexpr = BlockedLayout([4, 1], [1, 32], [1, 4], [0, 1])
    elements = gl.arange(0, 32, layout=SliceLayout(1, layout))
    threads = gl.arange(0, 128, layout=SliceLayout(0, layout))
    columns = (elements[:, None] // 4) * 512 + threads[None, :] * 4 + elements[:, None] % 4
    if EMBEDDING:
        token = gl.load(token_ids + row)
        hidden = gl.load(input_tensor + token * 4096 + columns).to(gl.float32)
    else:
        projected = gl.load(input_tensor + row * 4096 + columns).to(gl.float32)
        skip = gl.load(residual + row * 4096 + columns).to(gl.float32)
        # Eager aten.add materializes BF16 before the normalization reduction.
        hidden = (skip + projected).to(gl.bfloat16).to(gl.float32)
    partial_sum = gl.full((1, 128), 0, gl.float32, layout)
    for element in gl.static_range(32):
        value = gl.gather(hidden, gl.full((1, 128), element, gl.int32, layout), axis=0)
        partial_sum = gl.fma(value, value, partial_sum)
    square_sum = gl.sum(partial_sum.reshape((128,)), 0)
    inverse_rms = gl.rsqrt(square_sum / 4096 + 1e-5)
    weight_offsets = (columns // 512) * PACKED_ROW_STRIDE + columns % 512
    weight = gl.load(packed_parameters + weight_offsets).to(gl.float32)
    gl.store(normalized + row * 4096 + columns, (hidden * inverse_rms) * weight)


@triton.jit
def _copy_parameters(packed_parameters, outputs, SIZES: tl.constexpr,
                     OFFSETS: tl.constexpr, ROW_STRIDE: tl.constexpr,
                     BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    rank = tl.program_id(1)
    segment_start = 0
    for segment in tl.static_range(len(SIZES)):
        segment_tiles = triton.cdiv(SIZES[segment], BLOCK)
        if tile >= segment_start and tile < segment_start + segment_tiles:
            offsets = (tile - segment_start) * BLOCK + tl.arange(0, BLOCK)
            values = tl.load(packed_parameters + rank * ROW_STRIDE + OFFSETS[segment] + offsets,
                             offsets < SIZES[segment], other=0)
            tl.store(outputs[segment] + rank * SIZES[segment] + offsets,
                     values, offsets < SIZES[segment])
        segment_start += segment_tiles


def kernel_function(mm_216, add_60, wait_tensor_966):
    normalized = torch.empty((1, 8192, 4096), dtype=torch.bfloat16, device=mm_216.device)
    parameter_shapes = ((4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (4096,), (14336, 4096), (14336, 4096))
    parameters = tuple(torch.empty(shape, dtype=torch.bfloat16, device=wait_tensor_966.device)
                       for shape in parameter_shapes)
    _normalized_rows[(8192,)](mm_216, add_60, mm_216, wait_tensor_966, normalized,
                              27264000, False, num_warps=4, enable_fp_fusion=False)
    sizes = (2097152, 524288, 524288, 2097152, 512, 7340032, 7340032)
    offsets = (512, 2097664, 2621952, 3146240, 5243392, 5243904, 12583936)
    block = 8192
    _copy_parameters[(sum(triton.cdiv(size, block) for size in sizes), 8)](
        wait_tensor_966, parameters, sizes, offsets, 27264000, block, num_warps=4)
    return (normalized.view(8192, 4096), normalized.view(8192, 4096),
            normalized.view(8192, 4096), *parameters)


def roofline_workload(*inputs):
    """Minimum semantic traffic; the returned copies cannot alias the inputs."""
    copied_bytes = 318775296
    normalized_bytes = 8192 * 4096 * 2
    read_bytes = copied_bytes + 2 * normalized_bytes + 4096 * 2
    flops = 8192 * (4096 * 5 + 2)
    notes = ("Two BF16 residual inputs are read once; BF16-rounded sums stay in registers. "
             "Packed output slices are read and independently materialized once. The "
             "4096-element RMS weight can use cache. One native-order FP32 sum of squares "
             "and rsqrt per row, and two FP32 output multiplies per value.")
    write_bytes = copied_bytes + normalized_bytes
    return dict(read_bytes=read_bytes, write_bytes=write_bytes, flops=flops,
                special_ops=8192, working_set_bytes=read_bytes + write_bytes, notes=notes)
