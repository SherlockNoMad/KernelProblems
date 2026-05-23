import torch
import triton
import triton.language as tl


@triton.jit
def fused_feedforward_kernel(
    mm_4_ptr, mm_5_ptr, mm_662_ptr,
    out1_ptr, out2_ptr, out3_ptr,
    M, K: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_col_blocks = K // BLOCK_SIZE_COL
    row_block = pid // num_col_blocks
    col_block = pid % num_col_blocks

    row_start = row_block * BLOCK_SIZE_ROW
    col_start = col_block * BLOCK_SIZE_COL

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)

    offsets = row_offsets[:, None] * K + col_offsets[None, :]

    x = tl.load(mm_4_ptr + offsets).to(tl.float32)
    w3 = tl.load(mm_5_ptr + offsets).to(tl.float32)
    grad = tl.load(mm_662_ptr + offsets).to(tl.float32)

    sigmoid_x = tl.sigmoid(x)
    silu_x = x * sigmoid_x

    out1_val = silu_x * w3
    out2_val = grad * silu_x
    dsigmoid = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
    out3_val = (grad * w3) * dsigmoid

    tl.store(out1_ptr + offsets, out1_val.to(tl.bfloat16))
    tl.store(out2_ptr + offsets, out2_val.to(tl.bfloat16))
    tl.store(out3_ptr + offsets, out3_val.to(tl.bfloat16))


def kernel_function(mm_4, mm_5, mm_662):
    M, K = mm_4.shape
    dtype = mm_4.dtype
    device = mm_4.device

    out1 = torch.empty((M, K), dtype=dtype, device=device)
    out2_row = torch.empty((M, K), dtype=dtype, device=device)
    out3_row = torch.empty((M, K), dtype=dtype, device=device)

    BLOCK_SIZE_ROW = 1
    BLOCK_SIZE_COL = 8192
    
    num_blocks = (M // BLOCK_SIZE_ROW) * (K // BLOCK_SIZE_COL)

    fused_feedforward_kernel[(num_blocks,)](
        mm_4, mm_5, mm_662,
        out1, out2_row, out3_row,
        M, K,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        num_warps=8,
        num_stages=2,
    )

    out2 = out2_row.t()
    out3 = out3_row.t()

    return (out1, out2, out3)