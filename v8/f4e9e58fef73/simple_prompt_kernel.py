"""Bitwise-compatible RMSNorm specialized for the v8 BF16 [8192, 4096] case."""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def _rms_norm(
    input_ptr, weight_ptr, output_ptr, rstd_ptr, residual_ptr,
    ADD_RESIDUAL: gl.constexpr, STORE_RSTD: gl.constexpr,
):
    row = gl.program_id(0)
    # Native PyTorch uses 128 threads, each visiting eight vectors of four.
    # This layout retains those vectors in registers for the output pass.
    layout: gl.constexpr = BlockedLayout([4, 1], [1, 32], [1, 4], [0, 1])
    elements = gl.arange(0, 32, layout=SliceLayout(1, layout))
    threads = gl.arange(0, 128, layout=SliceLayout(0, layout))
    columns = elements[:, None] // 4 * 512 + threads[None, :] * 4 + elements[:, None] % 4
    values = gl.load(input_ptr + row * 4096 + columns, eviction_policy="evict_first").to(gl.float32)
    if ADD_RESIDUAL:
        residual = gl.load(residual_ptr + row * 4096 + columns, eviction_policy="evict_first").to(gl.float32)
        # aten.add materializes BF16 before native RMSNorm reads the residual.
        values = (values + residual).to(gl.bfloat16).to(gl.float32)

    square_sum = gl.full((1, 128), 0, gl.float32, layout)
    for element in gl.static_range(32):
        value = gl.gather(values, gl.full((1, 128), element, gl.int32, layout), axis=0)
        square_sum = gl.fma(value, value, square_sum)
    # Match native's warp reductions, then (warp0 + warp2) + (warp1 + warp3).
    square_sum = gl.sum(square_sum.reshape((128,)), 0)
    rstd = gl.rsqrt(square_sum * (1.0 / 4096.0) + 1e-5)
    if STORE_RSTD:
        gl.store(rstd_ptr + row, rstd)

    weight = gl.load(weight_ptr + columns, eviction_policy="evict_last").to(gl.float32)
    # Native applies both multiplications in FP32 and only then rounds to BF16.
    output = (values * rstd) * weight
    gl.store(output_ptr + row * 4096 + columns, output, cache_modifier=".cs")


def kernel_function(add_recomputed, _unsafe_view_428):
    output = torch.empty((1, 8192, 4096), dtype=add_recomputed.dtype, device=add_recomputed.device)
    rstd = torch.empty((8192,), dtype=torch.float32, device=add_recomputed.device)
    _rms_norm[(8192,)](
        add_recomputed, _unsafe_view_428, output, rstd, add_recomputed,
        ADD_RESIDUAL=False, STORE_RSTD=True, num_warps=4, enable_fp_fusion=False,
    )
    return output.view(8192, 4096), output.view(8192, 4096), rstd.view(1, 8192, 1)


def roofline_workload(add_recomputed, _unsafe_view_428):
    elements = add_recomputed.numel()
    read_bytes = add_recomputed.nbytes + _unsafe_view_428.nbytes
    write_bytes = add_recomputed.nbytes + 8192 * 4
    return {
        "read_bytes": read_bytes, "write_bytes": write_bytes,
        "flops": 4 * elements + 8192, "special_ops": 8192,
        "working_set_bytes": read_bytes + write_bytes,
        "notes": "One input read, one shared normalized output write, one FP32 rstd per row, "
                 "and one unique weight vector; the two normalized views alias. "
                 "FP32 square, sum, normalize, weight; one rsqrt per row. "
                 "Input tile is retained in registers.",
    }
