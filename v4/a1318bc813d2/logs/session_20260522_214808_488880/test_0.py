import torch

# Test for fused attention kernel with rotary positional embeddings
# Operations: _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy (repeated for q and k)
def test_kernel():
    """Test the kernel implementation."""
    try:
        from kernel import kernel_function
        # Sanity check: kernel should be callable and self-contained
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Device setup
        device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Create test data using exact specifications from problem description
        _conj_62 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device=device)
        _conj_63 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device=device)
        getitem_641 = torch.randn((33554432,), dtype=torch.bfloat16, device=device).as_strided([1, 32, 8192, 128], [33554432, 128, 4096, 1])
        getitem_642 = torch.randn((8388608,), dtype=torch.bfloat16, device=device).as_strided([1, 8, 8192, 128], [8388608, 128, 1024, 1])
        getitem_643 = torch.randn((8388608,), dtype=torch.bfloat16, device=device).as_strided([1, 8, 8192, 128], [8388608, 128, 1024, 1])

        # Create reference model for comparison
        from math import inf, nan
        
        class Model(torch.nn.Module):
            def forward(
                self,
                _conj_62,
                _conj_63,
                getitem_641,
                getitem_642,
                getitem_643,
            ):
                # Clone the complex tensors
                clone_default = torch.ops.aten.clone.default(_conj_62)
                clone_default_1 = torch.ops.aten.clone.default(_conj_63)

                # Process first tensor (queries)
                transpose_int = torch.ops.aten.transpose.int(getitem_641, 1, 2)
                _to_copy_default = torch.ops.aten._to_copy.default(transpose_int, dtype=torch.float32, layout=torch.strided, device=torch.device(type='cuda', index=0))
                reshape_default = torch.ops.aten.reshape.default(_to_copy_default, [1, 8192, 32, 64, 2])
                view_as_complex_default = torch.ops.aten.view_as_complex.default(reshape_default)
                mul_tensor = torch.ops.aten.mul.Tensor(view_as_complex_default, clone_default_1)
                view_as_real_default = torch.ops.aten.view_as_real.default(mul_tensor)
                reshape_default_1 = torch.ops.aten.reshape.default(view_as_real_default, [1, 8192, 32, 128])
                _to_copy_default_1 = torch.ops.aten._to_copy.default(reshape_default_1, dtype=torch.bfloat16, layout=torch.strided, device=torch.device(type='cuda', index=0))
                reshape_default_2 = torch.ops.aten.reshape.default(_to_copy_default_1, [1, 8192, 4096])
                reshape_default_3 = torch.ops.aten.reshape.default(reshape_default_2, [8192, 4096])
                t_default = torch.ops.aten.t.default(reshape_default_3)

                # Process second tensor (keys)
                transpose_int_1 = torch.ops.aten.transpose.int(getitem_642, 1, 2)
                _to_copy_default_2 = torch.ops.aten._to_copy.default(transpose_int_1, dtype=torch.float32, layout=torch.strided, device=torch.device(type='cuda', index=0))
                reshape_default_4 = torch.ops.aten.reshape.default(_to_copy_default_2, [1, 8192, 8, 64, 2])
                view_as_complex_default_1 = torch.ops.aten.view_as_complex.default(reshape_default_4)
                mul_tensor_1 = torch.ops.aten.mul.Tensor(view_as_complex_default_1, clone_default)
                view_as_real_default_1 = torch.ops.aten.view_as_real.default(mul_tensor_1)
                reshape_default_5 = torch.ops.aten.reshape.default(view_as_real_default_1, [1, 8192, 8, 128])
                _to_copy_default_3 = torch.ops.aten._to_copy.default(reshape_default_5, dtype=torch.bfloat16, layout=torch.strided, device=torch.device(type='cuda', index=0))
                reshape_default_6 = torch.ops.aten.reshape.default(_to_copy_default_3, [1, 8192, 1024])
                reshape_default_7 = torch.ops.aten.reshape.default(reshape_default_6, [8192, 1024])
                t_default_1 = torch.ops.aten.t.default(reshape_default_7)

                # Process third tensor (values - no rotary embedding)
                transpose_int_2 = torch.ops.aten.transpose.int(getitem_643, 1, 2)
                reshape_default_8 = torch.ops.aten.reshape.default(transpose_int_2, [1, 8192, 1024])
                reshape_default_9 = torch.ops.aten.reshape.default(reshape_default_8, [8192, 1024])
                t_default_2 = torch.ops.aten.t.default(reshape_default_9)
                
                return (t_default, t_default_1, t_default_2)

        # Compute reference output
        model = Model().to(device)
        with torch.no_grad():
            expected_outputs = model(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643)

        # Call kernel_function as a normal Python function
        result = kernel_function(_conj_62, _conj_63, getitem_641, getitem_642, getitem_643)

        # Verify result is a tuple of 3 tensors
        if not isinstance(result, tuple) or len(result) != 3:
            print(f"Expected tuple of 3 tensors, got: {type(result)} with length {len(result) if hasattr(result, '__len__') else 'N/A'}")
            return False

        # Check device consistency
        for i, (res_tensor, exp_tensor) in enumerate(zip(result, expected_outputs)):
            if not isinstance(res_tensor, torch.Tensor):
                print(f"Output {i} is not a tensor: {type(res_tensor)}")
                return False
            if res_tensor.device != exp_tensor.device:
                print(f"Output {i} device mismatch: expected {exp_tensor.device}, got {res_tensor.device}")
                return False

        # Verify results with detailed debugging on failure
        # Using looser tolerances for bfloat16 operations and complex arithmetic
        rtol, atol = 1e-2, 2e-2  # Looser tolerances for bfloat16 and complex operations
        
        for i, (res_tensor, exp_tensor) in enumerate(zip(result, expected_outputs)):
            if not torch.allclose(res_tensor, exp_tensor, rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH for output {i}:")
                print(f"Expected shape: {exp_tensor.shape}, dtype: {exp_tensor.dtype}")
                print(f"Result shape: {res_tensor.shape}, dtype: {res_tensor.dtype}")
                print(f"Expected (first few): {exp_tensor.flatten()[:10]}")
                print(f"Got (first few): {res_tensor.flatten()[:10]}")
                print(f"Max absolute difference: {torch.max(torch.abs(res_tensor - exp_tensor))}")
                print(f"Relative error: {torch.max(torch.abs((res_tensor - exp_tensor) / (exp_tensor + 1e-8)))}")
                return False

        print("All outputs match expected results!")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)