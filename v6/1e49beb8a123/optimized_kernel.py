import torch
import triton
import triton.language as tl


@triton.jit
def fused_embedding_multirow_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    total_rows,
    embed_dim,
    BLOCK_DIM: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_block_idx = tl.program_id(1)

    row_start = pid * ROWS_PER_BLOCK
    dim_offsets = dim_block_idx * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < embed_dim

    # Process multiple rows per block for better latency hiding
    for r in range(ROWS_PER_BLOCK):
        row_idx = row_start + r
        if row_idx < total_rows:
            embedding_idx = tl.load(indices_ptr + row_idx)
            src_base = embedding_idx * embed_dim
            dst_base = row_idx * embed_dim
            vals = tl.load(input_ptr + src_base + dim_offsets, mask=dim_mask, other=0.0)
            tl.store(output_ptr + dst_base + dim_offsets, vals, mask=dim_mask)


def kernel_function(wait_tensor_871, arg583_1):
    batch_size, seq_len = arg583_1.shape
    embed_dim = 4096

    output = torch.empty(
        (batch_size, seq_len, embed_dim),
        dtype=torch.bfloat16,
        device=wait_tensor_871.device,
    )

    flat_indices = arg583_1.reshape(-1)
    total_rows = flat_indices.shape[0]

    # Tuning: 4 rows per block increases ILP within each block,
    # allowing overlapping of memory requests from different rows.
    # BLOCK_DIM=1024 with 8 warps gives good balance of vectorization
    # and occupancy on H100.
    ROWS_PER_BLOCK = 4
    BLOCK_DIM = 1024
    num_warps = 8
    num_stages = 4

    dim_blocks = triton.cdiv(embed_dim, BLOCK_DIM)
    row_blocks = triton.cdiv(total_rows, ROWS_PER_BLOCK)

    grid = (row_blocks, dim_blocks)

    fused_embedding_multirow_kernel[grid](
        wait_tensor_871,
        flat_indices,
        output.view(-1, embed_dim),
        total_rows,
        embed_dim,
        BLOCK_DIM=BLOCK_DIM,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output