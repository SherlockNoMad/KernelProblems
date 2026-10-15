"""Fused RMS backward and ordered embedding gradients for the v8 fixed shapes.

The reduction order matches PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb
for the published contiguous BF16 [8192, 4096] inputs. BF16 additions, input
gradients, and weight gradients retain every eager rounding point. Embedding
contributions follow ascending token order in native groups of ten, followed
by one BF16 rounding and FP32 conversion. Integer atomics only assign metadata;
all floating-point reductions use deterministic ordered additions.
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


def _rms_backward(a, b, c, other, rstd, weight, residual):
    gradient = torch.empty((8192, 4096), device=a.device, dtype=torch.bfloat16)
    weight_gradient = torch.empty((4096,), device=a.device, dtype=torch.float32)
    weight_partials = torch.empty((32, 4096), device=a.device, dtype=torch.float32)
    # The saved rstd makes the two branches independent. Compute dx last so
    # its BF16 token gradients are available in cache for embedding backward.
    _weight_gradient_partial[(64, 32)](
        a, b, c, other, rstd, weight_partials, 1, 64, 3,
        num_warps=1, enable_fp_fusion=False,
    )
    _weight_gradient_reduce[(32,)](
        weight_partials, weight_gradient, 128,
        num_warps=4, enable_fp_fusion=False,
    )
    _rms_input_gradient[(8192,)](
        a, b, c, other, rstd, weight, residual, gradient, 1, 3,
        num_warps=4, maxnreg=128, enable_fp_fusion=False,
    )
    return gradient, weight_gradient



@triton.jit
def _embedding_initialize(heads,lengths,counters):
    position=tl.program_id(0)*1024+tl.arange(0,1024)
    tl.store(heads+position,8192,position<128256)
    tl.store(lengths+position,0,position<128256)
    if tl.program_id(0)==0:
        tl.store(counters+tl.arange(0,4),0)


@triton.jit
def _embedding_histogram(indices,heads,lengths):
    token=tl.program_id(0)*256+tl.arange(0,256)
    index=tl.load(indices+token)
    tl.atomic_min(heads+index,token,index>=0,sem='relaxed')
    tl.atomic_add(lengths+index,1,index>=0,sem='relaxed')


@triton.jit
def _embedding_count_add(a,b):
    return a+b


@triton.jit
def _embedding_compact_heads(indices,heads,unique_indices,counters):
    token=tl.arange(0,8192)
    index=tl.load(indices+token)
    head=tl.load(heads+index,index>=0,other=8192)
    first=(index>=0)&(head==token)
    rank=tl.associative_scan(first.to(tl.int32),0,_embedding_count_add)-1
    tl.store(unique_indices+rank,index.to(tl.int32),first)
    tl.store(counters+2,tl.sum(first.to(tl.int32),0))


@triton.jit
def _embedding_metadata(indices,heads,lengths,token_base,partial_base,tokens,unique_indices,
              chunk_start,chunk_length,counters,THRESHOLD:tl.constexpr):
    token=tl.program_id(0)
    index=tl.load(indices+token)
    if index>=0:
        head=tl.load(heads+index)
        if head==token:
            length=tl.load(lengths+index)
            # Short segments have fixed storage at their first token. Only
            # segments longer than 128 use the compact partial workspace.
            start=token*128
            if length>THRESHOLD:
                chunks=(length+9)//10
                base=tl.atomic_add(counters,chunks,sem='relaxed')
                tl.store(partial_base+index,base)
                start=8192*128+base*10
            tl.store(token_base+index,start)
            if length==1:
                tl.store(tokens+start,token)
            else:
                other=tl.arange(0,8192)
                matches=tl.load(indices+other)==index
                rank=tl.associative_scan(matches.to(tl.int32),0,_embedding_count_add)-1
                tl.store(tokens+start+rank,other,matches)
                if length>THRESHOLD:
                    base=(start-8192*128)//10
                    slot=base+rank//10
                    starts=matches & (rank%10==0)
                    tl.store(chunk_start+slot,start+rank,starts)
                    tl.store(chunk_length+slot,tl.minimum(10,length-rank),starts)


@triton.jit
def _embedding_partials(gradient,tokens,chunk_start,chunk_length,counters,partial,
              BLOCK:tl.constexpr):
    slot=tl.program_id(0)
    chunks=tl.load(counters)
    if slot<chunks:
        start=tl.load(chunk_start+slot)
        length=tl.load(chunk_length+slot)
        column=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        value=tl.full((BLOCK,),0,tl.float32)
        for item in tl.static_range(10):
            token=tl.load(tokens+start+item,item<length,other=0)
            value+=tl.load(gradient+token*4096+column,item<length,other=0).to(tl.float32)
        tl.store(partial+slot*4096+column,value)


@triton.jit
def _embedding_zero_empty(heads,output):
    index=tl.program_id(0)
    head=tl.load(heads+index)
    if head==8192:
        column=tl.arange(0,4096)
        tl.store(output+index*4096+column,0.0)


@g.jit
def _embedding_sparse_output(gradient,lengths,token_base,partial_base,tokens,unique_indices,
                      counters,partial,output,THRESHOLD:gl.constexpr,
                      BLOCK:gl.constexpr,PROGRAMS:gl.constexpr,TILE:gl.constexpr=32):
    layout:gl.constexpr=BlockedLayout([1,1],[1,32],[1,4],[1,0])
    cols=gl.program_id(1)*BLOCK+gl.arange(0,BLOCK,layout=SliceLayout(0,layout))
    chunk_offsets=gl.arange(0,TILE,layout=SliceLayout(1,layout))
    token_offsets=gl.arange(0,16,layout=SliceLayout(1,layout))
    unique_count=gl.load(counters+2)
    for unique in range(gl.program_id(0),unique_count,PROGRAMS):
        index=gl.load(unique_indices+unique)
        length=gl.load(lengths+index)
        value=gl.full((BLOCK,),0,gl.float32,SliceLayout(0,layout))
        if length==1:
            start=gl.load(token_base+index)
            token=gl.load(tokens+start)
            value=value+gl.load(gradient+token*4096+cols).to(gl.float32)
        elif length<=THRESHOLD:
            start=gl.load(token_base+index)
            for group_start in range(0,length,10):
                offsets=group_start+token_offsets
                valid=(token_offsets<10)&(offsets<length)
                token=gl.load(tokens+start+offsets,valid,other=0)
                values=gl.load(gradient+token[:,None]*4096+cols[None,:],valid[:,None],other=0).to(gl.float32)
                group=gl.full((1,BLOCK),0,gl.float32,layout)
                for item in gl.static_range(10):
                    group+=gl.gather(values,gl.full((1,BLOCK),item,gl.int32,layout),0)
                value+=gl.sum(group,0)
        else:
            start=gl.load(partial_base+index)
            chunks=(length+9)//10
            for first in range(0,chunks,TILE):
                chunk=first+chunk_offsets
                values=gl.load(partial+(start+chunk[:,None])*4096+cols[None,:],chunk[:,None]<chunks,other=0)
                group=value[None,:]
                for item in gl.static_range(TILE):
                    group+=gl.gather(values,gl.full((1,BLOCK),item,gl.int32,layout),0)
                value=gl.sum(group,0)
        gl.store(output+index*4096+cols,value.to(gl.bfloat16).to(gl.float32),cache_modifier='.cs')


def _embedding_workspace(gradient):
    device=gradient.device
    output=torch.empty((128256,4096),dtype=torch.float32,device=device)
    heads=torch.empty((128256,),dtype=torch.int32,device=device)
    lengths=torch.empty_like(heads)
    token_base=torch.empty_like(heads)
    partial_base=torch.empty_like(heads)
    tokens=torch.empty((8192*128+1024*10,),dtype=torch.int32,device=device)
    unique_indices=torch.empty((8192,),dtype=torch.int32,device=device)
    chunk_start=torch.empty((1024,),dtype=torch.int32,device=device)
    chunk_length=torch.empty_like(chunk_start)
    counters=torch.empty((4,),dtype=torch.int32,device=device)
    # With 8192 tokens and at least 129 per long segment, fewer than 1024
    # groups of ten can require partial storage, for every valid index pattern.
    partial=torch.empty((1024,4096),dtype=torch.float32,device=device)
    return output,heads,lengths,token_base,partial_base,tokens,unique_indices,chunk_start,chunk_length,counters,partial


def kernel_function(_fused_rms_norm_recomputed, mm_670, mm_672, mm_674,
                    embedding, _unsafe_view_432, add_220, arg583_1):
    workspace = _embedding_workspace(embedding)
    (output, heads, lengths, token_base, partial_base, tokens, unique_indices,
     chunk_start, chunk_length, counters, partial) = workspace
    _embedding_initialize[(126,)](heads, lengths, counters)
    _embedding_histogram[(32,)](arg583_1, heads, lengths)
    _embedding_compact_heads[(1,)](arg583_1, heads, unique_indices, counters, num_warps=8)
    _embedding_metadata[(8192,)](
        arg583_1, heads, lengths, token_base, partial_base, tokens, unique_indices,
        chunk_start, chunk_length, counters, 128, num_warps=8,
    )
    gradient, weight_gradient = _rms_backward(
        mm_670, mm_672, mm_674, embedding, _fused_rms_norm_recomputed[1],
        _unsafe_view_432, add_220,
    )
    _embedding_partials[(1024, 4)](
        gradient, tokens, chunk_start, chunk_length, counters, partial, 1024, num_warps=4,
    )
    _embedding_sparse_output[(1024, 8)](
        gradient, lengths, token_base, partial_base, tokens, unique_indices,
        counters, partial, output, 128, 512, 1024, 16, num_warps=4,
    )
    # Empty vocabulary rows are disjoint from the sparse stores above.
    _embedding_zero_empty[(128256,)](heads, output, num_warps=4)
    return weight_gradient, output


def roofline_workload(*inputs):
    elements = 8192 * 4096
    read_bytes = 5 * elements * 2 + 4096 * 2 + 8192 * 4 + 8192 * 8
    write_bytes = 128256 * 4096 * 4 + 4096 * 4
    return {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "flops": 18 * elements + 8192 * 128 + 4096 * 31,
        "special_ops": 0,
        "working_set_bytes": read_bytes + write_bytes,
        "notes": "Compulsory traffic counts five BF16 matrices, the used rstd and weight, "
                 "token indices, and two FP32 outputs; the unused RMS tuple output is "
                 "excluded. FLOPs include an upper bound of two FP32 additions per "
                 "embedding element (token and group accumulation). Collisions reduce "
                 "the group-add count. The implementation additionally rereads RMS "
                 "inputs for native-order weight reduction and uses BF16 token gradients, "
                 "integer metadata, and at most 1024x4096 FP32 embedding partials. "
                 "Every output row is written once; empty rows and occupied rows are "
                 "initialized by separate kernels. No floating-point atomic additions.",
    }
