import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    add_ptr, mm_ptr, weight_src_ptr, weight_col_offset, weight_row_stride,
    out_ptr, M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    row_off = row * N
    a = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    sumsq = tl.sum(x * x, axis=0)
    mean_sq = sumsq / N
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    # Weight is gathered from strided source: [8, 512] -> [4096]
    # row in [0,8), col in [0,512); flat idx = row*512 + col
    w_row = offs // 512
    w_col = offs % 512
    w_idx = w_row * weight_row_stride + weight_col_offset + w_col
    w = tl.load(weight_src_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w

    tl.store(out_ptr + row_off + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gather_strided_kernel(
    src_ptr, col_offset, sz, row_stride, dst_ptr, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    row = offs // sz
    col = offs % sz
    src_idx = row * row_stride + col_offset + col
    x = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _gather_multi_kernel(
    src_ptr,
    dst0_ptr, dst1_ptr, dst2_ptr, dst3_ptr, dst4_ptr, dst5_ptr, dst6_ptr,
    off0, off1, off2, off3, off4, off5, off6,
    sz0, sz1, sz2, sz3, sz4, sz5, sz6,
    total0, total1, total2, total3, total4, total5, total6,
    cum0, cum1, cum2, cum3, cum4, cum5, cum6,  # cumulative ends
    row_stride,
    grand_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < grand_total

    # Determine which slice each offset belongs to
    # Use cumulative bounds
    in0 = offs < cum0
    in1 = (offs >= cum0) & (offs < cum1)
    in2 = (offs >= cum1) & (offs < cum2)
    in3 = (offs >= cum2) & (offs < cum3)
    in4 = (offs >= cum3) & (offs < cum4)
    in5 = (offs >= cum4) & (offs < cum5)
    in6 = (offs >= cum5) & (offs < cum6)

    # local offset within the slice
    local0 = offs
    local1 = offs - cum0
    local2 = offs - cum1
    local3 = offs - cum2
    local4 = offs - cum3
    local5 = offs - cum4
    local6 = offs - cum5

    # Pick local offset, sz, col_offset based on which slice
    local = tl.where(in0, local0,
            tl.where(in1, local1,
            tl.where(in2, local2,
            tl.where(in3, local3,
            tl.where(in4, local4,
            tl.where(in5, local5, local6))))))
    sz = tl.where(in0, sz0,
         tl.where(in1, sz1,
         tl.where(in2, sz2,
         tl.where(in3, sz3,
         tl.where(in4, sz4,
         tl.where(in5, sz5, sz6))))))
    col_off = tl.where(in0, off0,
              tl.where(in1, off1,
              tl.where(in2, off2,
              tl.where(in3, off3,
              tl.where(in4, off4,
              tl.where(in5, off5, off6))))))

    row = local // sz
    col = local % sz
    src_idx = row * row_stride + col_off + col
    x = tl.load(src_ptr + src_idx, mask=mask)

    # Store to appropriate dst
    tl.store(dst0_ptr + local0, x, mask=mask & in0)
    tl.store(dst1_ptr + local1, x, mask=mask & in1)
    tl.store(dst2_ptr + local2, x, mask=mask & in2)
    tl.store(dst3_ptr + local3, x, mask=mask & in3)
    tl.store(dst4_ptr + local4, x, mask=mask & in4)
    tl.store(dst5_ptr + local5, x, mask=mask & in5)
    tl.store(dst6_ptr + local6, x, mask=mask & in6)


def kernel_function(mm_216, add_60, wait_tensor_966):
    M = 8192
    N = 4096
    eps = 1e-5
    device = mm_216.device

    ROW_STRIDE = 27264000
    splits = [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    col_offsets = [0]
    for s_ in splits:
        col_offsets.append(col_offsets[-1] + s_)

    add_view = add_60.view(M, N)
    mm_view = mm_216.view(M, N)

    # Stage A: fused add + rmsnorm, reading weight directly from strided source
    norm_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    BLOCK_N = 4096
    _add_rmsnorm_kernel[(M,)](
        add_view, mm_view,
        wait_tensor_966, col_offsets[0], ROW_STRIDE,
        norm_out, M, N, eps, BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    v1 = norm_out.view(M, N)
    v2 = norm_out.view(M, N)
    v3 = norm_out.view(M, N)

    # Allocate the 7 remaining outputs
    out_s1 = torch.empty((4096, 4096), dtype=torch.bfloat16, device=device)
    out_s2 = torch.empty((1024, 4096), dtype=torch.bfloat16, device=device)
    out_s3 = torch.empty((1024, 4096), dtype=torch.bfloat16, device=device)
    out_s4 = torch.empty((4096, 4096), dtype=torch.bfloat16, device=device)
    out_s5 = torch.empty(4096, dtype=torch.bfloat16, device=device)
    out_s6 = torch.empty((14336, 4096), dtype=torch.bfloat16, device=device)
    out_s7 = torch.empty((14336, 4096), dtype=torch.bfloat16, device=device)

    # Slice sizes (8 * split)
    totals = [8 * splits[i] for i in range(1, 9)]  # for slices 1..8

    # Fuse all gather operations into one kernel
    # 7 slices: indices 1..7 in our split list (slice 5 is small)
    # Use cumulative offsets
    t1, t2, t3, t4, t5, t6, t7 = totals[0], totals[1], totals[2], totals[3], totals[4], totals[5], totals[6]
    # Note: totals has 8 entries for slices 1..8 (output indices 1..7 plus slice 5)
    # Actually: splits[1..8] correspond to out_s1, out_s2, out_s3, out_s4, out_s5, out_s6, out_s7, and we need 8 outputs but there are 7 in output (s1..s7). Wait:
    # splits indices: 0=weight, 1=s1, 2=s2, 3=s3, 4=s4, 5=s5, 6=s6, 7=s7, 8=??
    # The original has 9 splits but only uses splits[0..7]. split[8] isn't used in outputs but exists.
    # Looking back: outputs are s1..s7 from splits[1..7], that's 7 outputs.

    t1 = 8 * splits[1]
    t2 = 8 * splits[2]
    t3 = 8 * splits[3]
    t4 = 8 * splits[4]
    t5 = 8 * splits[5]
    t6 = 8 * splits[6]
    t7 = 8 * splits[7]

    cum0 = t1
    cum1 = cum0 + t2
    cum2 = cum1 + t3
    cum3 = cum2 + t4
    cum4 = cum3 + t5
    cum5 = cum4 + t6
    cum6 = cum5 + t7
    grand = cum6

    BLOCK_SIZE = 2048
    grid = (triton.cdiv(grand, BLOCK_SIZE),)
    _gather_multi_kernel[grid](
        wait_tensor_966,
        out_s1, out_s2, out_s3, out_s4, out_s5, out_s6, out_s7,
        col_offsets[1], col_offsets[2], col_offsets[3], col_offsets[4],
        col_offsets[5], col_offsets[6], col_offsets[7],
        splits[1], splits[2], splits[3], splits[4], splits[5], splits[6], splits[7],
        t1, t2, t3, t4, t5, t6, t7,
        cum0, cum1, cum2, cum3, cum4, cum5, cum6,
        ROW_STRIDE,
        grand,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return (
        v1, v2, v3,
        out_s1, out_s2, out_s3,
        out_s4, out_s5,
        out_s6, out_s7,
    )