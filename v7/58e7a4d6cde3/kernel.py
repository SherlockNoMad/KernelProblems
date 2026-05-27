import torch
import triton
import triton.language as tl


@triton.jit
def _silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(mm_221):
    """Fused view + silu in a single Triton elementwise kernel.

    Stages fused:
      - reshape [8192, 14336] -> [1, 8192, 14336] (just a view, no data movement)
      - silu: x * sigmoid(x)
    """
    assert mm_221.is_cuda
    x = mm_221.contiguous()
    out_flat = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_kernel[grid](x, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    return out_flat.view(1, x.shape[0], x.shape[1])