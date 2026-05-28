import torch
import triton
import triton.language as tl


# ============================================================
# Fused stages implemented as Triton kernels:
#  1) add_rmsnorm_kernel: computes y = add_60 + mm_216 (viewed),
#     then applies RMSNorm along the last dim with weight from
#     wait_tensor_966[0:512] reinterpreted as bf16 [4096].
#  2) copy_slice_kernel: a generic elementwise copy that takes
#     a byte-offset into wait_tensor_966 (bf16) and copies N
#     bf16 elements into the destination output buffer. This
#     fuses the "view_dtype + clone + _unsafe_view" sequence into
#     a single contiguous copy.
# ============================================================


@triton.jit
def _add_rmsnorm_kernel(
    add_ptr,       # bf16 [M, N]
    mm_ptr,        # bf16 [M, N]
    weight_ptr,    # bf16 [N]
    out_ptr,       # bf16 [M, N]
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    row_off = row * N
    a = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(mm_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    # RMSNorm
    sumsq = tl.sum(x * x, axis=0)
    mean_sq = sumsq / N
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w

    tl.store(out_ptr + row_off + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_kernel(
    src_ptr,          # bf16 pointer (base of wait_tensor)
    src_offset,       # offset in bf16 elements
    dst_ptr,          # bf16 pointer (destination)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(src_ptr + src_offset + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


def _launch_copy(wait_tensor, src_offset_bf16, dst):
    n = dst.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _copy_kernel[grid](wait_tensor, src_offset_bf16, dst, n, BLOCK_SIZE=BLOCK_SIZE)


def kernel_function(mm_216, add_60, wait_tensor_966):
    """
    Fused implementation:
      - Stage A (fused add + rmsnorm with weight slice 0): computes
        getitem_1 = rmsnorm(add_60 + view(mm_216), weight=wait[0:4096 bf16])
      - Stage B (elementwise copies of slices): produces the other 7
        output buffers from contiguous slices of wait_tensor_966.

    The 3 view_default_{1,2,3} outputs share the same underlying buffer
    (getitem_1 reshaped). We expose them as views (PyTorch allowed for
    reshapes only — no compute).
    """
    assert mm_216.is_cuda and add_60.is_cuda and wait_tensor_966.is_cuda
    assert mm_216.dtype == torch.bfloat16
    assert add_60.dtype == torch.bfloat16
    assert wait_tensor_966.dtype == torch.bfloat16

    # Shapes
    M = 8192
    N = 4096
    eps = 1e-5

    # Reshape add_60 to [M, N] (view only; no compute)
    add_view = add_60.reshape(M, N)
    mm_view = mm_216.reshape(M, N)

    # ---- Stage A: fused add + rmsnorm ----
    # wait_tensor view as a flat bf16 array starting at element 0:
    # split_with_sizes is over a [8, -1] view in *bf16* elements? No —
    # wait_tensor is bf16 with 218112000 elems => /8 = 27264000 per row.
    # The split sizes [512, 2097152, 524288, 524288, 2097152, 512,
    # 7340032, 7340032, 7340032] are *bf16 element* counts along dim=1
    # of the [8, 27264000] view. Then each slice is reinterpreted via
    # view_dtype(bfloat16) — which is a no-op since it's already bf16.
    # Wait: view_dtype(getitem, bfloat16) on a bf16 tensor halves no
    # element count. Actually getitem is already bf16, so view_dtype to
    # bf16 is identity. But the expected output sizes are:
    #   slice 0: 512 -> [4096]  (mismatch! 512 != 4096)
    # That suggests the original wait_tensor is actually viewed as a
    # different dtype. Re-examine: wait_tensor_966 is bf16 with size
    # 218112000. view([8, -1]) gives [8, 27264000]. But then the splits
    # sum to 512+2097152+524288+524288+2097152+512+7340032*3 = ?
    s = 512 + 2097152 + 524288 + 524288 + 2097152 + 512 + 7340032 * 3
    # = 27264000. OK so the splits are in bf16 elements along dim 1.
    # Then view_dtype(slice, bfloat16): a bf16 -> bf16 cast: identity
    # (still 512 elements). But _unsafe_view to [4096] from 512 elems
    # would fail... Unless view_dtype here actually reinterprets bytes.
    # In aten, view.dtype reinterprets memory. For bf16 -> bf16 it's
    # identity. So how can 512 elements become [4096]?
    #
    # Re-reading: wait_tensor_966 is bf16 [218112000]. view([8, -1])
    # gives [8, 27264000]. Hmm but 218112000 / 8 = 27264000. Splits sum
    # to 27264000. Good.
    #
    # For slice 0 (512 bf16 elements), view as bf16 then view as [4096]
    # — 512 != 4096. So my interpretation must be wrong.
    #
    # Probably wait_tensor_966 is actually treated as a *byte buffer*
    # via view([8, -1]) on the bf16 tensor (so dim1 is in bf16 elems),
    # then view.dtype reinterprets to bf16 (no-op for bf16-source). But
    # the test expects _unsafe_view to [4096] to work, which requires
    # the slice to have 4096 elements. So the underlying assumption must
    # be that splits are in *bytes*, and view_dtype reinterprets bytes
    # to bf16 (so element count = bytes/2).
    #
    # Indeed: 512 bytes -> 256 bf16 elems? No, that's 256, not 4096.
    # Try: split is in bytes (uint8 view? but tensor is bf16). The
    # split_with_sizes is on a bf16 tensor with dim1 = 27264000 bf16
    # elements. The sizes 512, etc. are bf16 elements. To get 4096 bf16
    # elements from a 512-element bf16 slice, we'd need view_dtype to
    # *increase* element count, which only happens if going from a
    # larger dtype to bf16. So the original tensor must conceptually be
    # something larger... But it's bf16.
    #
    # Actually wait — let me recount: maybe view([8, -1]) yields more
    # than 27264000 in dim 1. 218112000/8 = 27264000.0 exactly. Splits
    # must sum to 27264000. Verified above.
    #
    # The only way slice0 (512 bf16 elems = 1024 bytes) becomes [4096]
    # is if it's reinterpreted as float8 (1 byte) -> 1024 elems. Still
    # not 4096. Or as uint4? Unlikely.
    #
    # Hmm — let me re-examine: perhaps wait_tensor is a *byte-packed*
    # buffer of a 4-byte type. If wait_tensor were float32 [218112000],
    # view to bf16 would give 2x elements. But it's declared bf16 in
    # test. Let me check whether view_dtype on bf16 -> bf16 is identity:
    # yes.
    #
    # Hold on — _unsafe_view doesn't validate element count strictly in
    # all builds; perhaps it just uses the new shape on the storage
    # respecting offset+size? No, _unsafe_view requires same numel.
    #
    # Let me recompute sizes table again carefully:
    #   [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    # Expected views: [4096], [4096,4096], [1024,4096], [1024,4096],
    # [4096,4096], [4096], [14336,4096], [14336,4096].
    # Element counts of views: 4096, 16777216, 4194304, 4194304,
    # 16777216, 4096, 58720256, 58720256.
    # Ratios (view_elems / slice_elems):
    #   4096/512 = 8
    #   16777216/2097152 = 8
    #   4194304/524288 = 8
    #   4194304/524288 = 8
    #   16777216/2097152 = 8
    #   4096/512 = 8
    #   58720256/7340032 = 8
    #   58720256/7340032 = 8
    # All factor 8! That means view_dtype is going from a 16-byte type
    # (e.g., int128/complex128?) to bf16 (2 bytes) → factor of 8.
    #
    # Or maybe wait_tensor_966 is declared as bf16 in the test but the
    # original problem expects it to be a different dtype. Looking at
    # the test code: `wait_tensor_966 = torch.randn([218112000],
    # dtype=torch.bfloat16, device=device)`. So it IS bf16.
    #
    # That means: 218112000 bf16 elems = 436224000 bytes.
    # view([8, -1]) on bf16 -> [8, 27264000]. Splits sum to 27264000.
    # But to get factor 8 in view_dtype to bf16... the source must be
    # 16 bytes per elem. But it's 2 bytes per elem.
    #
    # Unless view([8, -1]) on a bf16 with 218112000 elements means
    # something different. Wait — let's check if 218112000 is divisible
    # by 8: 218112000/8 = 27264000. Yes.
    #
    # The factor of 8 strongly suggests the wait_tensor is actually
    # being interpreted as uint8 (1 byte) and the splits are in bytes,
    # with view_dtype to bf16 dividing by 2. Then split sizes are byte
    # counts:
    #   512 bytes -> 256 bf16 elems. But view to [4096] needs 4096.
    # Still doesn't work.
    #
    # OR: wait_tensor is treated as int64 (8 bytes) -> bf16 (2 bytes)
    # gives factor 4. Not 8.
    #
    # The only thing giving factor 8: int64 view -> uint8? No, factor 8
    # = 16/2. So source dtype = 16 bytes (complex128).
    #
    # Hmm, this is strange. Let me look at the test again — maybe the
    # `view.dtype` op actually does something I don't know about. In
    # newer PyTorch, `torch.ops.aten.view.dtype` is just a reinterpret
    # cast. For bf16->bf16 it's identity.
    #
    # I think the test may be relying on a relaxed _unsafe_view that
    # actually treats the source as a different element count. Or
    # perhaps the test reference is just broken and we should match it.
    #
    # Actually wait: maybe the reference model errors out when run and
    # the test catches it? Let me look at the test flow: it runs Model
    # to get ref_outputs. If model errors, the test catches and returns
    # False. So we need the model to actually work.
    #
    # Let me reconsider: maybe view([8, -1]) on bf16 [218112000] doesn't
    # mean what I think. Hmm... 218112000 IS the right total.
    #
    # Oh! Maybe split sizes are intended to be interpreted differently.
    # Let me compute differently: if wait_tensor were viewed as bytes
    # (uint8) of size 436224000, then view([8, -1]) -> [8, 54528000].
    # Splits would need to sum to 54528000. 27264000*2=54528000. Hmm,
    # but split sizes given sum to 27264000.
    #
    # You know what, let me just trust the test: it expects to work.
    # Since this is hard to reason about without running, let me just
    # implement the most likely interpretation: that view_dtype on bf16
    # is identity, and the _unsafe_view following must match numel.
    # That would require slice 0 to have 4096 elems, not 512.
    #
    # Maybe I miscounted splits. Let me sum:
    # 512 + 2097152 = 2097664
    # +524288 = 2621952
    # +524288 = 3146240
    # +2097152 = 5243392
    # +512 = 5243904
    # +7340032 = 12583936
    # +7340032 = 19923968
    # +7340032 = 27264000. OK that's correct.
    #
    # Element counts of views: 4096+16777216+4194304+4194304+16777216
    # +4096+58720256+58720256+58720256. Let me sum:
    # 4096+16777216=16781312
    # +4194304=20975616
    # +4194304=25169920
    # +16777216=41947136
    # +4096=41951232
    # +58720256=100671488
    # +58720256=159391744
    # +58720256=218112000. THAT matches 218112000!
    #
    # So views sum to 218112000 bf16 elements, and slices sum to
    # 27264000. Ratio is exactly 8. So view_dtype DOES multiply by 8.
    #
    # That means wait_tensor_966 in storage is 218112000 bf16 elements
    # = 436224000 bytes. But after view([8, -1]) the row size in
    # *split* terms is 27264000/8 = 3408000... no wait, view([8,-1]) on
    # bf16 [218112000] gives [8, 27264000] (in bf16 elements). Splits
    # sum to 27264000 (bf16 elems). Then view_dtype changes element
    # count by factor 8. That's only possible if the source dtype is
    # interpreted as larger.
    #
    # OHHHHH. I bet `view.dtype` here is a NO-OP on dtype but the
    # underlying tensor's element size is being computed differently
    # because... hmm.
    #
    # Actually wait: maybe the [8, -1] view is being done on the bf16
    # tensor *but the strides/storage size are based on different elem
    # size*. No.
    #
    # Let me try another theory: maybe `view.dtype(getitem, bfloat16)`
    # when the source IS bfloat16 returns the same tensor, but then
    # clone with contiguous_format... that's still same numel.
    #
    # I cannot reconcile this with my understanding of PyTorch. Let me
    # just try running it: I'll write the implementation assuming the
    # splits ARE in the final element counts (4096, 16777216, etc.),
    # ignoring the literal `split_with_sizes` numbers, and using offsets
    # that match the view shapes.
    #
    # Actually, here's an idea: maybe view([8, -1]) on bf16 of size N
    # gives [8, N/8] in bf16 elems. Then split [512, 2097152, ...] on
    # dim 1 of size N/8=27264000. Then getitem is bf16 of shape
    # [8, 512]. view_dtype to bf16 = identity, shape [8, 512].
    # clone(contiguous) = [8, 512] = 4096 elems. THEN _unsafe_view to
    # [4096] = OK!
    #
    # YES! That's it! I forgot: view([8, -1]) creates an [8, X] tensor.
    # The split is on dim 1, but getitem is still 2D with [8, slice_sz].
    # Numel = 8 * slice_sz! So slice 0 has 8*512 = 4096 elems!
    #
    # Great, mystery solved.

    # So slice i has numel = 8 * split_sizes[i], and is laid out as
    # [8, split_sizes[i]] — NOT contiguous in the original [218112000]
    # buffer (because dim 1 is strided by 27264000 in the original).
    # Wait no — view([8, -1]) on contiguous [218112000] gives [8, 27264000]
    # where dim 0 stride is 27264000 and dim 1 stride is 1. Then
    # split on dim 1 yields slices of shape [8, sz] with strides
    # (27264000, 1). clone(contiguous) makes them contiguous [8, sz].
    # Then _unsafe_view to flat shape.
    #
    # So we need to gather strided data: for each output slice, read
    # 8 rows of `sz` elements at offsets row*27264000 + col_offset,
    # and write them contiguously.

    device = mm_216.device

    ROW_STRIDE = 27264000  # bf16 elements per row in view([8, -1])
    splits = [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032]
    col_offsets = [0]
    for s_ in splits:
        col_offsets.append(col_offsets[-1] + s_)

    # Slice 0: weight for rmsnorm, shape [8*512] = [4096]
    weight = torch.empty(4096, dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[0], splits[0], ROW_STRIDE, weight)

    # Output for rmsnorm
    norm_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    BLOCK_N = 4096  # = N, fits in one block
    grid_norm = (M,)
    _add_rmsnorm_kernel[grid_norm](
        add_view, mm_view, weight, norm_out, M, N, eps, BLOCK_N=BLOCK_N
    )

    # view_default_{1,2,3}: reshape norm_out to [8192, 4096] (same buffer)
    v1 = norm_out.view(M, N)
    v2 = norm_out.view(M, N)
    v3 = norm_out.view(M, N)

    # Slice 1: [4096, 4096] = 16777216 elements
    out_s1 = torch.empty((4096, 4096), dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[1], splits[1], ROW_STRIDE, out_s1)

    # Slice 2: [1024, 4096]
    out_s2 = torch.empty((1024, 4096), dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[2], splits[2], ROW_STRIDE, out_s2)

    # Slice 3: [1024, 4096]
    out_s3 = torch.empty((1024, 4096), dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[3], splits[3], ROW_STRIDE, out_s3)

    # Slice 4: [4096, 4096]
    out_s4 = torch.empty((4096, 4096), dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[4], splits[4], ROW_STRIDE, out_s4)

    # Slice 5: [4096]
    out_s5 = torch.empty(4096, dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[5], splits[5], ROW_STRIDE, out_s5)

    # Slice 6: [14336, 4096]
    out_s6 = torch.empty((14336, 4096), dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[6], splits[6], ROW_STRIDE, out_s6)

    # Slice 7: [14336, 4096]
    out_s7 = torch.empty((14336, 4096), dtype=torch.bfloat16, device=device)
    _gather_strided(wait_tensor_966, col_offsets[7], splits[7], ROW_STRIDE, out_s7)

    return (
        v1, v2, v3,
        out_s1, out_s2, out_s3,
        out_s4, out_s5,
        out_s6, out_s7,
    )


@triton.jit
def _gather_strided_kernel(
    src_ptr,          # bf16 base
    col_offset,       # column offset in bf16 elems
    sz,               # slice size per row (bf16 elems)
    row_stride,       # bf16 elems between rows in source
    dst_ptr,          # bf16 dest (contiguous, shape [8, sz] flattened)
    n_rows,           # = 8
    total,            # = 8 * sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    # row = offs // sz, col = offs % sz
    row = offs // sz
    col = offs % sz
    src_idx = row * row_stride + col_offset + col
    x = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


def _gather_strided(src, col_offset, sz, row_stride, dst):
    """Gathers an [8, sz] strided slice from src into contiguous dst."""
    total = 8 * sz
    assert dst.numel() == total, f"dst numel {dst.numel()} != {total}"
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    _gather_strided_kernel[grid](
        src, col_offset, sz, row_stride, dst, 8, total, BLOCK_SIZE=BLOCK_SIZE
    )