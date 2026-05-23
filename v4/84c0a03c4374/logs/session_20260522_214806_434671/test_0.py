import torch
import torch.nn as nn

# Test for fused region: _fused_rms_norm -> add -> _fused_rms_norm_backward -> add -> _to_copy
# This kernel fuses RMS normalization operations with additions and type conversion
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

        # Create test data using EXACT specifications from problem description
        # Using the same input generation as provided in get_inputs()
        getitem_1624 = torch.randn((190848512,), dtype=torch.bfloat16, device=device).as_strided([8, 512], [27264000, 1])
        add_62_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device=device)
        mm_230 = torch.randn([8192, 4096], dtype=torch.bfloat16, device=device)
        mm_232 = torch.randn([8192, 4096], dtype=torch.bfloat16, device=device)
        getitem_420 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device=device)

        # Create reference model for comparison
        class ReferenceModel(torch.nn.Module):
            def forward(self, getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420):
                # Follow the exact computation sequence from the provided model
                view_dtype = torch.ops.aten.view.dtype(getitem_1624, torch.bfloat16)
                clone_default = torch.ops.aten.clone.default(view_dtype, memory_format=torch.contiguous_format)
                _unsafe_view_default = torch.ops.aten._unsafe_view.default(clone_default, [4096])
                
                _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_62_recomputed, [4096], _unsafe_view_default, 1e-05)
                getitem = _fused_rms_norm_default[0]
                
                reshape_default = torch.ops.aten.reshape.default(getitem, [8192, 4096])
                reshape_default_1 = torch.ops.aten.reshape.default(getitem, [8192, 4096])
                getitem_1625 = _fused_rms_norm_default[1]
                
                reshape_default_2 = torch.ops.aten.reshape.default(mm_230, [1, 8192, 4096])
                reshape_default_3 = torch.ops.aten.reshape.default(mm_232, [1, 8192, 4096])
                
                add_tensor = torch.ops.aten.add.Tensor(reshape_default_2, reshape_default_3)
                
                _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(
                    add_tensor, add_62_recomputed, [4096], getitem_1625, _unsafe_view_default, [True, True])
                
                getitem_1626 = _fused_rms_norm_backward_default[0]
                add_tensor_1 = torch.ops.aten.add.Tensor(getitem_420, getitem_1626)
                getitem_421 = _fused_rms_norm_backward_default[1]
                _to_copy_default = torch.ops.aten._to_copy.default(getitem_421, dtype=torch.float32)
                
                return (reshape_default, reshape_default_1, add_tensor_1, _to_copy_default)

        # Compute reference output
        ref_model = ReferenceModel().to(device)
        with torch.no_grad():
            expected_outputs = ref_model(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420)

        # Call kernel_function as a normal Python function
        result_outputs = kernel_function(getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420)

        # Verify outputs are tuples/lists of the same length
        if not (isinstance(result_outputs, (tuple, list)) and isinstance(expected_outputs, (tuple, list))):
            print(f"Output type mismatch: expected tuple/list, got {type(result_outputs)} and {type(expected_outputs)}")
            return False
        
        if len(result_outputs) != len(expected_outputs):
            print(f"Output count mismatch: expected {len(expected_outputs)}, got {len(result_outputs)}")
            return False

        # Check each output tensor
        for i, (result, expected) in enumerate(zip(result_outputs, expected_outputs)):
            if not isinstance(result, torch.Tensor) or not isinstance(expected, torch.Tensor):
                print(f"Output {i}: expected tensor, got {type(result)} and {type(expected)}")
                return False

            # Device check
            if result.device != expected.device:
                print(f"Output {i}: device mismatch - expected {expected.device}, got {result.device}")
                return False

            # Shape check
            if result.shape != expected.shape:
                print(f"Output {i}: shape mismatch - expected {expected.shape}, got {result.shape}")
                return False

            # Dtype check
            if result.dtype != expected.dtype:
                print(f"Output {i}: dtype mismatch - expected {expected.dtype}, got {result.dtype}")
                return False

            # Numerical comparison with appropriate tolerances for bfloat16
            # Using looser tolerances due to bfloat16 precision and complex fused operations
            rtol, atol = 1e-2, 2e-2  # Adjusted for bfloat16 and fused operations
            if not torch.allclose(result, expected, rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH for output {i}:")
                print(f"Input shapes: getitem_1624={getitem_1624.shape}, add_62_recomputed={add_62_recomputed.shape}")
                print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first few): {expected.flatten()[:10]}")
                print(f"Got (first few): {result.flatten()[:10]}")
                print(f"Max absolute difference: {torch.max(torch.abs(result - expected))}")
                print(f"Relative error: {torch.max(torch.abs((result - expected) / (expected + 1e-8)))}")
                return False

        print("All outputs match reference implementation!")
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