import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_add_kernel(
    mm_ptr, add_ptr, out0_ptr, out1_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: computes out0 = add + mm and out1 = add + mm in a single pass.
    
    Fusion rationale: both outputs share the same computation (add_61 + view(mm_220)),
    so we load each input once and write both outputs, saving memory bandwidth.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    mm = tl.load(mm_ptr + offsets, mask=mask)
    add = tl.load(add_ptr + offsets, mask=mask)
    
    result = add + mm
    
    tl.store(out0_ptr + offsets, result, mask=mask)
    tl.store(out1_ptr + offsets, result, mask=mask)


def kernel_function(mm_220, add_61):
    """Fused add+add wrapper.
    
    Fuses two identical elementwise additions (add_61 + mm_220.view(1,8192,4096))
    into a single Triton kernel pass that produces both outputs.
    """
    assert mm_220.is_cuda and add_61.is_cuda
    assert mm_220.numel() == add_61.numel()
    
    # Reshape mm_220 to match add_61 shape (no data movement, just view)
    view0 = mm_220.view(1, *mm_220.shape) if mm_220.dim() == 2 else mm_220
    # Ensure shapes are compatible
    assert view0.shape == add_61.shape, f"Shape mismatch: {view0.shape} vs {add_61.shape}"
    
    out0 = torch.empty_like(add_61)
    out1 = torch.empty_like(add_61)
    
    n_elements = add_61.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _fused_add_add_kernel[grid](
        view0, add_61, out0, out1,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out0, out1)