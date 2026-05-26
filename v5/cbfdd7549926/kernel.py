import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    emb_ptr, mm3_ptr, sum_ptr, N_ELEM,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEM
    a = tl.load(emb_ptr + offs, mask=mask, other=0.0)
    b = tl.load(mm3_ptr + offs, mask=mask, other=0.0)
    tl.store(sum_ptr + offs, a + b, mask=mask)


@triton.jit
def _fwd_rms_kernel(
    x_ptr, weight_ptr, out_ptr, rstd_ptr,
    N: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _bwd_dx_kernel(
    mm664_ptr, mm666_ptr, x_ptr, weight_ptr, rstd_ptr,
    add218_ptr, out_ptr, dweight_partial_ptr,
    M, N: tl.constexpr, BLOCK: tl.constexpr,
):
    """
    RMS norm backward:
      grad_out = mm664 + mm666
      x_hat = x * rstd
      dy_w = grad_out * w
      c = sum(dy_w * x_hat) / N
      dx = rstd * (dy_w - x_hat * c)
      out = add218 + dx  (cast dx to bf16 before adding to match reference)
      dweight_partial[row,:] = grad_out * x_hat
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    go1 = tl.load(mm664_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    go2 = tl.load(mm666_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # match reference: add_tensor_1 = view2 + view3 done in bf16, then passed to backward
    grad_out_bf16 = (go1 + go2)
    # cast back to bf16 to match the bf16 add_tensor_1 input to backward
    grad_out = grad_out_bf16  # will round via bf16 cast below
    # Round to bf16 then back to fp32 to emulate the bf16 add_tensor_1
    grad_out = grad_out.to(tl.bfloat16).to(tl.float32)

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dy_w = grad_out * w
    c = tl.sum(dy_w * x_hat, axis=0) / N
    dx = rstd * (dy_w - x_hat * c)

    add218 = tl.load(add218_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # cast dx to bf16 first (reference returns bf16 getitem_789, then adds to add_218)
    dx_bf16 = dx.to(tl.bfloat16).to(tl.float32)
    out = add218 + dx_bf16
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

    dw_partial = grad_out * x_hat
    tl.store(dweight_partial_ptr + row * N + cols, dw_partial, mask=mask)


@triton.jit
def _reduce_dweight_kernel(
    dw_partial_ptr, dw_out_ptr,
    M, N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        r = start + rows
        mask = r < M
        vals = tl.load(dw_partial_ptr + r * N + col, mask=mask, other=0.0)
        acc += vals
    total = tl.sum(acc, axis=0)
    tl.store(dw_out_ptr + col, total)


def kernel_function(mm_3, embedding, getitem_787, mm_664, mm_666, add_218):
    """
    Fused implementation:
      - add_tensor = embedding + mm_3
      - RMS norm forward producing normalized output + rstd
      - grad_out = mm_664 + mm_666 (bf16)
      - RMS norm backward producing dx (bf16) and dweight (fp32)
      - add_tensor_2 = add_218 + dx
    
    Note: Kept as separate kernels for add, fwd-norm, bwd-dx, reduce-dweight because
    the backward depends on rstd from forward (cross-row dependency through reduction)
    and dweight reduction is cross-row. Could fuse add+fwd but kept separate for clarity;
    forward and backward must be separate phases since backward needs completed rstd.
    """
    assert mm_3.is_cuda and embedding.is_cuda
    M = 8192
    N = 4096

    weight = getitem_787.contiguous().view(-1)[:N].contiguous()
    assert weight.numel() == N
    assert weight.dtype == torch.bfloat16

    fwd_out = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_3.device)
    rstd = torch.empty((1, M, 1), dtype=torch.float32, device=mm_3.device)
    add_tensor = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_3.device)

    n_elem = M * N
    BLOCK = 1024
    grid_add = (triton.cdiv(n_elem, BLOCK),)
    _add_kernel[grid_add](embedding, mm_3, add_tensor, n_elem, BLOCK=BLOCK)

    BLOCK_N = triton.next_power_of_2(N)
    _fwd_rms_kernel[(M,)](
        add_tensor, weight, fwd_out, rstd,
        N=N, eps=1e-5, BLOCK=BLOCK_N,
        num_warps=8,
    )

    out_add218 = torch.empty((1, M, N), dtype=torch.bfloat16, device=mm_3.device)
    dw_partial = torch.empty((M, N), dtype=torch.float32, device=mm_3.device)

    _bwd_dx_kernel[(M,)](
        mm_664, mm_666, add_tensor, weight, rstd,
        add_218, out_add218, dw_partial,
        M, N=N, BLOCK=BLOCK_N,
        num_warps=8,
    )

    dweight = torch.empty((N,), dtype=torch.float32, device=mm_3.device)
    BLOCK_M_RED = 256
    _reduce_dweight_kernel[(N,)](
        dw_partial, dweight, M, N=N, BLOCK_M=BLOCK_M_RED,
    )

    view_default = fwd_out.view(M, N)
    view_default_1 = fwd_out.view(M, N)

    return (view_default, view_default_1, out_add218, dweight)