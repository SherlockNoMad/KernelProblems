"""Fused RMS backward preserving the native CUDA floating-point operation order.

The reduction order matches PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb
for the published contiguous BF16 [8192, 4096] inputs. BF16 additions, input
gradients, and weight gradients retain every eager rounding point.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.gluon as g
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


# MODE 0 sums two incoming gradients; MODE 1 sums three. MODE 2 instead
# adds three activations and recomputes rstd, with X providing the direct dy.
# The [32,128] layout retains the native 128 lanes and eight vec4 loads.
# Both accumulation and the final left product require explicit FMA. The right
# product rounds before its sign bit flips; arithmetic 0-x could lose -0.
@g.jit
def _rms_input_gradient(A, B, C, X, R, W, RESIDUAL, OUT, MODE: gl.constexpr, CACHE: gl.constexpr=0):
    row = gl.program_id(0)
    layout: gl.constexpr = BlockedLayout([4, 1], [1, 32], [1, 4], [0, 1])
    elems = gl.arange(0, 32, layout=SliceLayout(1, layout))
    threads = gl.arange(0, 128, layout=SliceLayout(0, layout))
    cols = elems[:, None] // 4 * 512 + threads[None, :] * 4 + elems[:, None] % 4
    offsets = row * 4096 + cols
    ordinary: gl.constexpr = 'evict_first' if CACHE == 1 or CACHE == 2 else ''
    selected: gl.constexpr = 'evict_last' if CACHE == 1 else ordinary
    modifier: gl.constexpr = '.cg' if CACHE == 3 else ''
    a = gl.load(A + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(gl.float32)
    b = gl.load(B + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(gl.float32)
    summed = (a + b).to(gl.bfloat16).to(gl.float32)
    if MODE != 0:
        c = gl.load(C + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(gl.float32)
        summed = (summed + c).to(gl.bfloat16).to(gl.float32)
    other = gl.load(X + offsets, eviction_policy=selected, cache_modifier=modifier).to(gl.float32)
    if MODE == 2:
        x = summed
        dy = other
        sigma = gl.full((1, 128), 0, gl.float32, layout)
        for i in gl.static_range(32):
            value = gl.gather(x, gl.full((1, 128), i, gl.int32, layout), 0)
            sigma = gl.fma(value, value, sigma)
        rstd = gl.rsqrt(gl.sum(sigma.reshape((128,)), 0) / 4096.0 + 1e-05)
        gl.store(R + row, rstd)
    else:
        x = other
        dy = summed
        rstd = gl.load(R + row)
    w = gl.load(W + cols).to(gl.float32)
    contribution = dy * w * x
    stat = gl.full((1, 128), 0, gl.float32, layout)
    for i in gl.static_range(32):
        value = gl.gather(contribution, gl.full((1, 128), i, gl.int32, layout), 0)
        stat = gl.fma(value, rstd, stat)
    stat = gl.sum(stat.reshape((128,)), 0)
    right = x * rstd * stat
    negative_right = (right.to(gl.int32, bitcast=True) ^ -2147483648).to(gl.float32, bitcast=True)
    dx = gl.fma(4096.0 * w, dy, negative_right) * (rstd / 4096.0)
    dx = dx.to(gl.bfloat16)
    if MODE != 2:
        residual = gl.load(RESIDUAL + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(gl.float32)
        dx = (dx.to(gl.float32) + residual).to(gl.bfloat16)
    gl.store(OUT + offsets, dx, eviction_policy=ordinary)


# Each of the native 32 row groups accumulates 256 terms in a fixed order.
# Tile-loop unrolling exposes loads while preserving that FMA dependency chain.
@triton.jit
def _weight_gradient_partial(A, B, C, X, R, PARTIAL, MODE: tl.constexpr, BLOCK: tl.constexpr=64, CACHE: tl.constexpr=0):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = tl.program_id(1)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    ordinary: tl.constexpr = 'evict_first' if CACHE == 1 or CACHE == 2 else ''
    selected: tl.constexpr = 'evict_last' if CACHE == 1 else ordinary
    modifier: tl.constexpr = '.cg' if CACHE == 3 else ''
    for tile in tl.range(32, num_stages=1, loop_unroll_factor=4):
        for k in tl.static_range(8):
            row = tile * 256 + group * 8 + k
            offsets = row * 4096 + cols
            a = tl.load(A + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(tl.float32)
            b = tl.load(B + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(tl.float32)
            summed = (a + b).to(tl.bfloat16).to(tl.float32)
            if MODE != 0:
                c = tl.load(C + offsets, eviction_policy=ordinary, cache_modifier=modifier).to(tl.float32)
                summed = (summed + c).to(tl.bfloat16).to(tl.float32)
            other = tl.load(X + offsets, eviction_policy=selected, cache_modifier=modifier).to(tl.float32)
            rstd = tl.load(R + row)
            acc = tl.fma(summed * other, rstd, acc)
    tl.store(PARTIAL + group * 4096 + cols, acc)


# Native gamma backward uses XOR 16,8,4,2,1. A generic two-dimensional sum
# may choose a different tree and change the final BF16 result near a tie.
@triton.jit
def _weight_gradient_reduce(PARTIAL, OUT, BLOCK: tl.constexpr=128):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    groups = tl.arange(0, 32)
    partial = tl.load(PARTIAL + groups[:, None] * 4096 + cols[None, :])
    for delta in tl.static_range(4, -1, -1):
        shuffled = tl.gather(partial, tl.broadcast_to((groups ^ 1 << delta)[:, None], (32, BLOCK)), 0)
        partial = partial + shuffled
    total = tl.gather(partial, tl.full((1, BLOCK), 0, tl.int32), 0).reshape((BLOCK,))
    tl.store(OUT + cols, total.to(tl.bfloat16).to(tl.float32))


def _rms_backward(a, b, c, other, rstd, weight, residual, mode):
    gradient = torch.empty((8192, 4096), device=a.device, dtype=torch.bfloat16)
    weight_gradient = torch.empty((4096,), device=a.device, dtype=torch.float32)
    weight_partials = torch.empty((32, 4096), device=a.device, dtype=torch.float32)
    if mode == 2:
        rstd = torch.empty((8192,), device=a.device, dtype=torch.float32)
    input_cache = 3 if mode == 1 else 2
    weight_cache = 1 if mode == 0 else 3 if mode == 1 else 2
    _rms_input_gradient[(8192,)](
        a, b, c, other, rstd, weight, residual, gradient, mode, input_cache,
        num_warps=4, maxnreg=128, enable_fp_fusion=False,
    )
    _weight_gradient_partial[(64, 32)](
        a, b, c, other, rstd, weight_partials, mode, 64, weight_cache,
        num_warps=1, enable_fp_fusion=False,
    )
    _weight_gradient_reduce[(32,)](
        weight_partials, weight_gradient, 128,
        num_warps=4, enable_fp_fusion=False,
    )
    return gradient, weight_gradient


def kernel_function(_fused_rms_norm_2_recomputed, mm_656, mm_658, mm_660,
                    add_1, _unsafe_view_450, add_215):
    return _rms_backward(mm_656, mm_658, mm_660, add_1,
                         _fused_rms_norm_2_recomputed[1], _unsafe_view_450, add_215, 1)


def roofline_workload(*inputs):
    elements = 8192 * 4096
    read_bytes = 5 * elements * 2 + 4096 * 2 + 32768
    write_bytes = elements * 2 + 4096 * 4
    return {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "flops": 16 * elements + 8192 * 128 + 4096 * 31,
        "special_ops": 0,
        "working_set_bytes": read_bytes + write_bytes,
        "notes": "Minimum compulsory traffic counts the used BF16 matrices, weight, "
                 "external rstd where present, and two outputs. The unused RMS "
                 "tuple output is excluded. FP32 FMA counts as two operations. "
                 "The implementation additionally rereads inputs for the fixed "
                 "weight-gradient reduction and uses 32x4096 FP32 partials; "
                 "the minimum-traffic roofline does not hide that overhead.",
    }
