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


def kernel_function(mm_220, add_61, _unsafe_view_981):
    output = torch.empty((1, 8192, 4096), dtype=mm_220.dtype, device=mm_220.device)
    _rms_norm[(8192,)](
        add_61, _unsafe_view_981, output, output, mm_220,
        ADD_RESIDUAL=True, STORE_RSTD=False, num_warps=4, enable_fp_fusion=False,
    )
    return output.view(8192, 4096), output.view(8192, 4096)


def roofline_workload(mm_220, add_61, _unsafe_view_981):
    elements = add_61.numel()
    read_bytes = mm_220.nbytes + add_61.nbytes + _unsafe_view_981.nbytes
    write_bytes = add_61.nbytes
    return {
        "read_bytes": read_bytes, "write_bytes": write_bytes,
        "flops": 5 * elements + 8192, "special_ops": 8192,
        "working_set_bytes": read_bytes + write_bytes,
        "notes": "Two residual input reads, one shared output write, and one unique "
                 "weight vector; both returned views alias. The BF16-rounded residual "
                 "stays in registers. Residual add plus FP32 square, sum, normalize, "
                 "weight; one rsqrt per row.",
    }
