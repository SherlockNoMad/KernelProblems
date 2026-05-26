import torch
import triton
import triton.language as tl

WAIT_ROW_STRIDE = 65667584
SPLIT0_SIZE = 512
SPLIT1_SIZE = 65667072
NUM_CHUNKS = 8
G2_ROWS = 128256
G2_COLS = 4096
G2_ROWS_PER_CHUNK = SPLIT1_SIZE // G2_COLS
N_FEATURES = 4096
N_ROWS = 8192


@triton.jit
def _extract_g2_kernel(src_ptr, dst_ptr,
                       WAIT_ROW_STRIDE: tl.constexpr,
                       SPLIT0_SIZE: tl.constexpr,
                       G2_COLS: tl.constexpr,
                       G2_ROWS_PER_CHUNK: tl.constexpr,
                       N_TOTAL,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_TOTAL
    row = offs // G2_COLS
    col = offs % G2_COLS
    chunk = row // G2_ROWS_PER_CHUNK
    within_row = row % G2_ROWS_PER_CHUNK
    src_idx = chunk * WAIT_ROW_STRIDE + SPLIT0_SIZE + within_row * G2_COLS + col
    x = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _fused_add_rmsnorm_with_weight_extract_kernel(
        add_ptr, mm_ptr, wait_ptr, out_ptr,
        WAIT_ROW_STRIDE: tl.constexpr,
        SPLIT0_SIZE: tl.constexpr,
        N_COLS: tl.constexpr,
        eps,
        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    row_off = row * N_COLS
    a = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    # Inline weight extraction from wait_tensor
    chunk = offs // SPLIT0_SIZE
    within = offs % SPLIT0_SIZE
    w_idx = chunk * WAIT_ROW_STRIDE + within
    w = tl.load(wait_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)

    sq = x * x
    var = tl.sum(sq, axis=0) / N_COLS
    rstd = 1.0 / tl.sqrt(var + eps)

    y = x * rstd * w

    tl.store(out_ptr + row_off + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_223, add_62, wait_tensor_969):
    assert mm_223.dtype == torch.bfloat16
    assert add_62.dtype == torch.bfloat16
    assert wait_tensor_969.dtype == torch.bfloat16

    device = mm_223.device

    mm_c = mm_223.contiguous()
    add_c = add_62.contiguous()
    wait_c = wait_tensor_969.contiguous()

    # Stage 2: extract split[1] into [128256, 4096]
    g2_buf = torch.empty((G2_ROWS, G2_COLS), dtype=torch.bfloat16, device=device)
    BLOCK_G2 = 2048
    n_total = G2_ROWS * G2_COLS
    grid_g2 = (triton.cdiv(n_total, BLOCK_G2),)
    _extract_g2_kernel[grid_g2](
        wait_c, g2_buf,
        WAIT_ROW_STRIDE=WAIT_ROW_STRIDE,
        SPLIT0_SIZE=SPLIT0_SIZE,
        G2_COLS=G2_COLS,
        G2_ROWS_PER_CHUNK=G2_ROWS_PER_CHUNK,
        N_TOTAL=n_total,
        BLOCK=BLOCK_G2,
    )
    t_default = g2_buf.t()

    # Stage 3: fused add + RMSNorm with inline weight extraction
    out_view = torch.empty((N_ROWS, N_FEATURES), dtype=torch.bfloat16, device=device)
    BLOCK_C = 4096
    grid_rn = (N_ROWS,)
    _fused_add_rmsnorm_with_weight_extract_kernel[grid_rn](
        add_c, mm_c, wait_c, out_view,
        WAIT_ROW_STRIDE=WAIT_ROW_STRIDE,
        SPLIT0_SIZE=SPLIT0_SIZE,
        N_COLS=N_FEATURES,
        eps=1e-5,
        BLOCK=BLOCK_C,
        num_warps=8,
    )

    return (out_view, t_default)