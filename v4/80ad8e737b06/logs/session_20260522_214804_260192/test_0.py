import torch
import torch.nn as nn

# Test for fused loss kernel: _to_copy -> _log_softmax -> nll_loss_forward -> div -> ones_like -> div -> nll_loss_backward -> _log_softmax_backward_data -> _to_copy
def test_kernel():
    """Test the fused loss kernel implementation."""
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
        # Based on the Model class: arg584_1 is target indices, mm_224 is logits
        batch_size = 8192
        vocab_size = 128256
        
        # Target indices - using valid range [0, vocab_size)
        target_indices = torch.randint(0, vocab_size, [1, batch_size], dtype=torch.int64, device=device)
        # Logits - using bfloat16 as specified
        logits = torch.randn([batch_size, vocab_size], dtype=torch.bfloat16, device=device)
        
        # Ensure we have some non-ignored targets (not -100)
        target_indices = target_indices.clamp(0, vocab_size - 1)

        # Create reference model to compute expected output
        class ReferenceModel(torch.nn.Module):
            def forward(self, arg584_1, mm_224):
                # Following the exact computation from the provided Model
                reshape_default = torch.ops.aten.reshape.default(arg584_1, [batch_size])
                reshape_default_1 = torch.ops.aten.reshape.default(mm_224, [1, batch_size, vocab_size])
                reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, [batch_size, vocab_size])
                
                _to_copy_default = torch.ops.aten._to_copy.default(reshape_default_2, dtype=torch.float32)
                _log_softmax_default = torch.ops.aten._log_softmax.default(_to_copy_default, 1, False)
                nll_loss_forward_default = torch.ops.aten.nll_loss_forward.default(_log_softmax_default, reshape_default, None, 2, -100)
                
                getitem = nll_loss_forward_default[0]
                div_tensor = torch.ops.aten.div.Tensor(getitem, 8192.0)
                ones_like_default = torch.ops.aten.ones_like.default(div_tensor, pin_memory=False, memory_format=torch.preserve_format)
                div_tensor_1 = torch.ops.aten.div.Tensor(ones_like_default, 8192.0)
                getitem_1 = nll_loss_forward_default[1]
                
                nll_loss_backward_default = torch.ops.aten.nll_loss_backward.default(div_tensor_1, _log_softmax_default, reshape_default, None, 2, -100, getitem_1)
                _log_softmax_backward_data_default = torch.ops.aten._log_softmax_backward_data.default(nll_loss_backward_default, _log_softmax_default, 1, torch.float32)
                _to_copy_default_1 = torch.ops.aten._to_copy.default(_log_softmax_backward_data_default, dtype=torch.bfloat16, layout=torch.strided, device=device)
                
                reshape_default_3 = torch.ops.aten.reshape.default(_to_copy_default_1, [1, batch_size, vocab_size])
                reshape_default_4 = torch.ops.aten.reshape.default(reshape_default_3, [batch_size, vocab_size])
                t_default = torch.ops.aten.t.default(reshape_default_4)
                return t_default

        # Compute reference output
        ref_model = ReferenceModel().to(device)
        with torch.no_grad():
            expected = ref_model(target_indices, logits)

        # Call kernel_function as a normal Python function
        result = kernel_function(target_indices, logits)

        # Device checks
        if not isinstance(result, torch.Tensor):
            print(f"Expected tensor output, got {type(result)}")
            return False
            
        if result.device != logits.device:
            print(f"Device mismatch: result on {result.device}, expected {logits.device}")
            return False

        # Shape and dtype checks
        if result.shape != expected.shape:
            print(f"Shape mismatch: got {result.shape}, expected {expected.shape}")
            return False
            
        if result.dtype != expected.dtype:
            print(f"Dtype mismatch: got {result.dtype}, expected {expected.dtype}")
            return False

        # Verify results with detailed debugging on failure
        # Using looser tolerances for bfloat16 and large accumulation dimension
        rtol, atol = 1e-2, 2e-2  # Looser tolerances for bfloat16 and large vocab_size accumulation
        
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print(f"NUMERICAL MISMATCH:")
            print(f"Target indices shape: {target_indices.shape}, dtype: {target_indices.dtype}")
            print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            print(f"Expected (first few): {expected.flatten()[:10]}")
            print(f"Got (first few): {result.flatten()[:10]}")
            print(f"Max absolute difference: {torch.max(torch.abs(result - expected))}")
            print(f"Relative error: {torch.max(torch.abs((result - expected) / (expected + 1e-8)))}")
            print(f"Target sample: {target_indices.flatten()[:10]}")
            print(f"Logits sample: {logits.flatten()[:10]}")
            return False

        print("Test passed successfully!")
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