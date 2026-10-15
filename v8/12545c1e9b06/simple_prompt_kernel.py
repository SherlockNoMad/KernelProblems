"""Native-order BF16 embedding/RMSNorm, fused with independent parameter copies."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def _embedding_rms_row(embedding_weights, token_ids, packed_parameters, normalized, row):
    # Match aten's 128-thread vec4 accumulation, warp tree, and FP32 epilogue.
    layout: gl.constexpr = BlockedLayout([4, 1], [1, 32], [1, 4], [0, 1])
    elements = gl.arange(0, 32, layout=SliceLayout(1, layout))
    threads = gl.arange(0, 128, layout=SliceLayout(0, layout))
    columns = (elements[:, None] // 4) * 512 + threads[None, :] * 4 + elements[:, None] % 4
    token = gl.load(token_ids + row)
    hidden = gl.load(embedding_weights + token * 4096 + columns).to(gl.float32)
    partial_sum = gl.full((1, 128), 0, gl.float32, layout)
    for element in gl.static_range(32):
        value = gl.gather(hidden, gl.full((1, 128), element, gl.int32, layout), axis=0)
        partial_sum = gl.fma(value, value, partial_sum)
    square_sum = gl.sum(partial_sum.reshape((128,)), 0)
    inverse_rms = gl.rsqrt(square_sum / 4096 + 1e-5)
    weight_offsets = (columns // 512) * 27264000 + columns % 512
    weight = gl.load(packed_parameters + weight_offsets).to(gl.float32)
    gl.store(normalized + row * 4096 + columns, (hidden * inverse_rms) * weight)


@gluon.jit
def _embedding_norm_and_copies(embedding_weights, token_ids, packed_parameters,
                               normalized, parameters):
    # Interleave one normalization CTA with three copy CTAs to overlap the
    # native reduction chain with the much larger independent copy workload.
    work = gl.program_id(0)
    group = work // 4
    group_lane = work % 4
    if group_lane == 0:
        if group < 8192:
            _embedding_rms_row(embedding_weights, token_ids, packed_parameters, normalized, group)
    else:
        copy_tile = group * 3 + group_lane - 1
        sizes: gl.constexpr = (2097152, 524288, 524288, 2097152, 512, 7340032)
        source_offsets: gl.constexpr = (512, 2097664, 2621952, 3146240, 5243392, 5243904)
        rank = copy_tile // 3073
        tile = copy_tile % 3073
        if rank < 8:
            segment_start = 0
            for segment in gl.static_range(len(sizes)):
                segment_tiles = triton.cdiv(sizes[segment], 4096)
                if tile >= segment_start and tile < segment_start + segment_tiles:
                    offsets = (tile - segment_start) * 4096 + gl.arange(
                        0, 4096, layout=BlockedLayout([8], [32], [4], [0]))
                    values = gl.load(packed_parameters + rank * 27264000 + source_offsets[segment] + offsets,
                                     offsets < sizes[segment], other=0)
                    gl.store(parameters[segment] + rank * sizes[segment] + offsets,
                             values, offsets < sizes[segment])
                segment_start += segment_tiles


def kernel_function(wait_tensor_871, arg583_1, wait_tensor_873):
    normalized = torch.empty((1, 8192, 4096), dtype=torch.bfloat16, device=wait_tensor_871.device)
    parameter_shapes = ((4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (4096,), (14336, 4096))
    parameters = tuple(torch.empty(shape, dtype=torch.bfloat16, device=wait_tensor_873.device)
                       for shape in parameter_shapes)
    _embedding_norm_and_copies[(32780,)](
        wait_tensor_871, arg583_1, wait_tensor_873, normalized, parameters,
        num_warps=4, enable_fp_fusion=False)
    return (normalized.view(8192, 4096), normalized.view(8192, 4096),
            normalized.view(8192, 4096), *parameters)


def roofline_workload(*inputs):
    """Minimum semantic traffic; the returned copies cannot alias the inputs."""
    copied_bytes = 201334784
    normalized_bytes = 8192 * 4096 * 2
    embedding_bytes = torch.unique(inputs[1]).numel() * 4096 * 2
    read_bytes = copied_bytes + embedding_bytes + inputs[1].numel() * 8 + 4096 * 2
    flops = 8192 * (4096 * 4 + 2)
    notes = ("Each unique indexed embedding row is read once in the HBM bound; repeated rows and "
             "the gathered 4096-element RMS weight can use cache. Every packed output is a "
             "materialized independent allocation. One native-order FP32 sum of squares and "
             "rsqrt per row, and two FP32 output multiplies per value.")
    write_bytes = copied_bytes + normalized_bytes
    return dict(read_bytes=read_bytes, write_bytes=write_bytes, flops=flops,
                special_ops=8192, working_set_bytes=read_bytes + write_bytes, notes=notes)
