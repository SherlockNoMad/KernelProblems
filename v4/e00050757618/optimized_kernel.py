import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_to_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output_data = input_data.to(tl.float32)
    tl.store(output_ptr + offsets, output_data, mask=mask)


def kernel_function(input_tensor):
    assert input_tensor.dtype == torch.bfloat16
    assert input_tensor.is_cuda

    # Ensure contiguous for coalesced access
    input_tensor = input_tensor.contiguous()

    n_elements = input_tensor.numel()
    output = torch.empty(input_tensor.shape, dtype=torch.float32, device=input_tensor.device)

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    fused_to_copy_kernel[grid](
        input_tensor,
        output,
        n_elements,
    )

    return output