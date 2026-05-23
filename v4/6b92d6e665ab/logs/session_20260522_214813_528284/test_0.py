import torch

# Test for fused attention kernel with rotary positional embeddings
# The kernel performs RoPE operations on query/key tensors with complex arithmetic
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
        # Based on the Model class forward signature and get_inputs function
        arg586_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int32, device=device)
        arg582_1 = torch.randn([8192, 64], dtype=torch.complex64, device=device)
        mm_217 = torch.randn([8192, 4096], dtype=torch.bfloat16, device=device)
        mm_218 = torch.randn([8192, 1024], dtype=torch.bfloat16, device=device)

        # Create reference model to compute expected output
        class Model(torch.nn.Module):
            def forward(
                self,
                arg586_1,
                arg582_1,
                mm_217,
                mm_218,
            ):
                # Following the exact operations from the problem description
                squeeze_dim = torch.ops.aten.squeeze.dim(arg586_1, 0)
                index_tensor = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim])
                reshape_default = torch.ops.aten.reshape.default(index_tensor, [1, 8192, 1, 64])

                # Process query tensor (mm_217)
                reshape_default_1 = torch.ops.aten.reshape.default(mm_217, [1, 8192, 4096])
                reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, [1, 8192, -1, 128])
                _to_copy_default = torch.ops.aten._to_copy.default(reshape_default_2, dtype=torch.float32)
                reshape_default_3 = torch.ops.aten.reshape.default(_to_copy_default, [1, 8192, 32, -1, 2])
                view_as_complex_default = torch.ops.aten.view_as_complex.default(reshape_default_3)
                mul_tensor = torch.ops.aten.mul.Tensor(view_as_complex_default, reshape_default)
                view_as_real_default = torch.ops.aten.view_as_real.default(mul_tensor)
                reshape_default_4 = torch.ops.aten.reshape.default(view_as_real_default, [1, 8192, 32, 128])
                _to_copy_default_1 = torch.ops.aten._to_copy.default(reshape_default_4, dtype=torch.bfloat16, layout=torch.strided, device=device)
                transpose_int = torch.ops.aten.transpose.int(_to_copy_default_1, 1, 2)

                # Process key tensor (mm_218)
                reshape_default_5 = torch.ops.aten.reshape.default(mm_218, [1, 8192, 1024])
                reshape_default_6 = torch.ops.aten.reshape.default(reshape_default_5, [1, 8192, -1, 128])
                _to_copy_default_2 = torch.ops.aten._to_copy.default(reshape_default_6, dtype=torch.float32)
                reshape_default_7 = torch.ops.aten.reshape.default(_to_copy_default_2, [1, 8192, 8, -1, 2])
                view_as_complex_default_1 = torch.ops.aten.view_as_complex.default(reshape_default_7)
                mul_tensor_1 = torch.ops.aten.mul.Tensor(view_as_complex_default_1, reshape_default)
                view_as_real_default_1 = torch.ops.aten.view_as_real.default(mul_tensor_1)
                reshape_default_8 = torch.ops.aten.reshape.default(view_as_real_default_1, [1, 8192, 8, 128])
                _to_copy_default_3 = torch.ops.aten._to_copy.default(reshape_default_8, dtype=torch.bfloat16, layout=torch.strided, device=device)
                transpose_int_1 = torch.ops.aten.transpose.int(_to_copy_default_3, 1, 2)

                # Second pass operations (duplicated processing)
                reshape_default_9 = torch.ops.aten.reshape.default(mm_217, [1, 8192, 4096])
                reshape_default_10 = torch.ops.aten.reshape.default(mm_218, [1, 8192, 1024])
                reshape_default_11 = torch.ops.aten.reshape.default(reshape_default_9, [1, 8192, -1, 128])
                reshape_default_12 = torch.ops.aten.reshape.default(reshape_default_10, [1, 8192, -1, 128])
                _to_copy_default_4 = torch.ops.aten._to_copy.default(reshape_default_11, dtype=torch.float32)
                reshape_default_13 = torch.ops.aten.reshape.default(_to_copy_default_4, [1, 8192, 32, -1, 2])
                view_as_complex_default_2 = torch.ops.aten.view_as_complex.default(reshape_default_13)
                _to_copy_default_5 = torch.ops.aten._to_copy.default(reshape_default_12, dtype=torch.float32)
                reshape_default_14 = torch.ops.aten.reshape.default(_to_copy_default_5, [1, 8192, 8, -1, 2])
                view_as_complex_default_3 = torch.ops.aten.view_as_complex.default(reshape_default_14)
                
                squeeze_dim_1 = torch.ops.aten.squeeze.dim(arg586_1, 0)
                index_tensor_1 = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_1])
                reshape_default_15 = torch.ops.aten.reshape.default(index_tensor_1, [1, 8192, 1, 64])
                
                mul_tensor_2 = torch.ops.aten.mul.Tensor(view_as_complex_default_2, reshape_default_15)
                view_as_real_default_2 = torch.ops.aten.view_as_real.default(mul_tensor_2)
                reshape_default_16 = torch.ops.aten.reshape.default(view_as_real_default_2, [1, 8192, 32, 128])
                mul_tensor_3 = torch.ops.aten.mul.Tensor(view_as_complex_default_3, reshape_default_15)
                view_as_real_default_3 = torch.ops.aten.view_as_real.default(mul_tensor_3)
                reshape_default_17 = torch.ops.aten.reshape.default(view_as_real_default_3, [1, 8192, 8, 128])
                
                _to_copy_default_6 = torch.ops.aten._to_copy.default(reshape_default_16, dtype=torch.bfloat16, layout=torch.strided, device=device)
                _to_copy_default_7 = torch.ops.aten._to_copy.default(reshape_default_17, dtype=torch.bfloat16, layout=torch.strided, device=device)
                transpose_int_2 = torch.ops.aten.transpose.int(_to_copy_default_6, 1, 2)
                transpose_int_3 = torch.ops.aten.transpose.int(_to_copy_default_7, 1, 2)
                
                return (transpose_int, transpose_int_1, transpose_int_2, transpose_int_3)

        # Compute reference output
        model = Model()
        expected_outputs = model(arg586_1, arg582_1, mm_217, mm_218)

        # Call kernel_function as a normal Python function
        result = kernel_function(arg586_1, arg582_1, mm_217, mm_218)

        # Verify result is a tuple with 4 outputs
        if not isinstance(result, tuple) or len(result) != 4:
            print(f"Expected tuple of 4 outputs, got {type(result)} with length {len(result) if hasattr(result, '__len__') else 'N/A'}")
            return False

        # Check each output tensor
        for i, (res, exp) in enumerate(zip(result, expected_outputs)):
            if not isinstance(res, torch.Tensor):
                print(f"Output {i} is not a tensor: {type(res)}")
                return False
            
            # Device check
            if res.device != exp.device:
                print(f"Output {i} device mismatch: expected {exp.device}, got {res.device}")
                return False

            # Shape check
            if res.shape != exp.shape:
                print(f"Output {i} shape mismatch: expected {exp.shape}, got {res.shape}")
                return False

            # Dtype check
            if res.dtype != exp.dtype:
                print(f"Output {i} dtype mismatch: expected {exp.dtype}, got {res.dtype}")
                return False

            # Numerical comparison - using looser tolerances for bfloat16 and complex operations
            if not torch.allclose(res, exp, rtol=1e-2, atol=2e-2):
                print(f"NUMERICAL MISMATCH in output {i}:")
                print(f"Input shapes: arg586_1={arg586_1.shape}, arg582_1={arg582_1.shape}, mm_217={mm_217.shape}, mm_218={mm_218.shape}")
                print(f"Expected shape: {exp.shape}, dtype: {exp.dtype}")
                print(f"Result shape: {res.shape}, dtype: {res.dtype}")
                print(f"Expected (first few): {exp.flatten()[:10]}")
                print(f"Got (first few): {res.flatten()[:10]}")
                print(f"Max absolute difference: {torch.max(torch.abs(res - exp))}")
                print(f"Relative error: {torch.max(torch.abs((res - exp) / (exp + 1e-8)))}")
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