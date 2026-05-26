import torch
import triton
import triton.language as tl

# Constants from the problem
# wait_tensor: [525340672] = 8 * 65667584
# Split on dim=1 of view(8, 65667584): sizes [512, 65667072]
# - getitem (size 512 per row, 8 rows) -> reshape to [4096] = rms_norm weight
# - getitem_2 (size 65667072 per row, 8 rows) -> view [128256, 4096], transpose to [4096, 128256]
# add_tensor = add_62 + mm_223.view(1,8192,4096)
# rms_norm(add_tensor, weight, eps=1e-5).view(8192, 4096)

WAIT_ROW_STRIDE = 65667584      # original row size after view(8, -1)
SPLIT0_SIZE = 512                # first split size per row
SPLIT1_SIZE = 65667072           # second split size per row
NUM_CHUNKS = 8
G2_TOTAL = NUM_CHUNKS * SPLIT1_SIZE  # 525336576
G2_ROWS = 128256                 # 8 * 16032
G2_COLS = 4096
G2_ROWS_PER_CHUNK = SPLIT1_SIZE // G2_COLS  # 16032
N_FEATURES = 4096
N_ROWS = 8192


@triton.jit
def _extract_weight_kernel(src_ptr, dst_ptr,
                           WAIT_ROW_STRIDE: tl.constexpr,
                           SPLIT0_SIZE: tl.constexpr,
                           N_FEATURES: tl.constexpr,
                           BLOCK: tl.constexpr):
    """Extract weight (split[0]) from wait_tensor into contiguous [4096]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_FEATURES
    chunk = offs // SPLIT0_SIZE
    within = offs % SPLIT0_SIZE
    src_idx = chunk * WAIT_ROW_STRIDE + within
    x = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _extract_g2_kernel(src_ptr, dst_ptr,
                       WAIT_ROW_STRIDE: tl.constexpr,
                       SPLIT0_SIZE: tl.constexpr,
                       SPLIT1_SIZE: tl.constexpr,
                       G2_COLS: tl.constexpr,
                       G2_ROWS_PER_CHUNK: tl.constexpr,
                       N_TOTAL,
                       BLOCK: tl.constexpr):
    """Extract split[1] from wait_tensor into contiguous [128256, 4096]."""
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
def _fused_add_rmsnorm_kernel(add_ptr, mm_ptr, w_ptr, out_ptr,
                              N_COLS: tl.constexpr,
                              eps,
                              BLOCK: tl.constexpr):
    """Fused: out = rms_norm(add_62 + mm_223, weight, eps), one row per program."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    row_off = row * N_COLS
    a = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    # Mean of squares
    sq = x * x
    var = tl.sum(sq, axis=0) / N_COLS
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_ptr + row_off + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_223, add_62, wait_tensor_969):
    """
    Fused pipeline:
      1) extract weight (split[0]) from wait_tensor -> [4096]
      2) extract split[1] from wait_tensor -> [128256, 4096] (then .t() for output)
      3) fused (add_62 + mm_223) + rms_norm with extracted weight -> [8192, 4096]

    Fusion notes:
      - Stage 3 fuses the elementwise add and RMSNorm (mean-square reduction,
        rsqrt, weight scaling) in a single Triton kernel per row.
      - Stages 1 and 2 cannot be merged with stage 3 because the rms_norm
        weight (stage 1) is produced from wait_tensor and must be ready
        before stage 3; stage 2 produces an independent output. They run as
        separate Triton kernels.
    """
    assert mm_223.is_cuda and add_62.is_cuda and wait_tensor_969.is_cuda
    assert mm_223.dtype == torch.bfloat16
    assert add_62.dtype == torch.bfloat16
    assert wait_tensor_969.dtype == torch.bfloat16
    assert mm_223.shape == (N_ROWS, N_FEATURES)
    assert add_62.numel() == N_ROWS * N_FEATURES
    assert wait_tensor_969.numel() == NUM_CHUNKS * WAIT_ROW_STRIDE

    device = mm_223.device

    # Ensure contiguity for simple linear addressing
    mm_c = mm_223.contiguous()
    add_c = add_62.contiguous()
    wait_c = wait_tensor_969.contiguous()

    # Stage 1: extract weight [4096]
    weight = torch.empty((N_FEATURES,), dtype=torch.bfloat16, device=device)
    BLOCK_W = 1024
    grid_w = (triton.cdiv(N_FEATURES, BLOCK_W),)
    _extract_weight_kernel[grid_w](
        wait_c, weight,
        WAIT_ROW_STRIDE=WAIT_ROW_STRIDE,
        SPLIT0_SIZE=SPLIT0_SIZE,
        N_FEATURES=N_FEATURES,
        BLOCK=BLOCK_W,
    )

    # Stage 2: extract split[1] into [128256, 4096], then return transpose (metadata-only)
    g2_buf = torch.empty((G2_ROWS, G2_COLS), dtype=torch.bfloat16, device=device)
    BLOCK_G2 = 2048
    n_total = G2_ROWS * G2_COLS
    grid_g2 = (triton.cdiv(n_total, BLOCK_G2),)
    _extract_g2_kernel[grid_g2](
        wait_c, g2_buf,
        WAIT_ROW_STRIDE=WAIT_ROW_STRIDE,
        SPLIT0_SIZE=SPLIT0_SIZE,
        SPLIT1_SIZE=SPLIT1_SIZE,
        G2_COLS=G2_COLS,
        G2_ROWS_PER_CHUNK=G2_ROWS_PER_CHUNK,
        N_TOTAL=n_total,
        BLOCK=BLOCK_G2,
    )
    t_default = g2_buf.t()  # metadata-only transpose, no compute

    # Stage 3: fused add + RMSNorm
    out_view = torch.empty((N_ROWS, N_FEATURES), dtype=torch.bfloat16, device=device)
    BLOCK_C = 4096  # power-of-two >= N_FEATURES (==4096)
    grid_rn = (N_ROWS,)
    _fused_add_rmsnorm_kernel[grid_rn](
        add_c, mm_c, weight, out_view,
        N_COLS=N_FEATURES,
        eps=1e-5,
        BLOCK=BLOCK_C,
        num_warps=8,
    )

    return (out_view, t_default)