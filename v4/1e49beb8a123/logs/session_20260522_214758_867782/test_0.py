import torch
import torch.nn as nn

# Test for fused region (tok_embeddings): split_with_sizes -> embedding
# This kernel fuses view operations with embedding lookup
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
        # Based on the shapes in the problem:
        # wait_tensor_871: "bf16[525336576][1]cuda:0"
        # arg583_1: "i64[1, 8192][8192, 1]cuda:0"
        wait_tensor_871 = torch.randn([525336576], dtype=torch.bfloat16, device=device)
        arg583_1 = torch.randint(0, 128256, [1, 8192], dtype=torch.int64, device=device)  # vocab size is 128256 based on view_default_1 shape

        # Create reference model for comparison
        class ReferenceModel(torch.nn.Module):
            def forward(self, wait_tensor_871, arg583_1):
                # Reference implementation from problem description
                view_default = torch.ops.aten.view.default(wait_tensor_871, [8, -1])
                split_with_sizes_default = torch.ops.aten.split_with_sizes.default(view_default, [65667072], 1)
                getitem = split_with_sizes_default[0]
                view_dtype = torch.ops.aten.view.dtype(getitem, torch.bfloat16)
                view_default_1 = torch.ops.aten.view.default(view_dtype, [128256, 4096])
                
                # Create embedding table with correct dimensions
                # From the output shape "bf16[1, 8192, 4096]", vocab_size=128256, embed_dim=4096
                embedding_weight = view_default_1  # This acts as the embedding table
                embedding_default = torch.ops.aten.embedding.default(embedding_weight, arg583_1)
                return embedding_default

        # Compute reference output
        ref_model = ReferenceModel().to(device)
        with torch.no_grad():
            expected = ref_model(wait_tensor_871.clone(), arg583_1.clone())

        # Call kernel_function as a normal Python function
        result = kernel_function(wait_tensor_871, arg583_1)

        # Device check
        if isinstance(result, torch.Tensor) and result.device != wait_tensor_871.device:
            print(f"Device mismatch: result on {result.device}, input on {wait_tensor_871.device}")
            return False

        # Shape and dtype checks
        if result.shape != expected.shape:
            print(f"Shape mismatch: expected {expected.shape}, got {result.shape}")
            return False
        
        if result.dtype != expected.dtype:
            print(f"Dtype mismatch: expected {expected.dtype}, got {result.dtype}")
            return False

        # Verify results with detailed debugging on failure
        # Using looser tolerances for bfloat16 due to lower precision
        if not torch.allclose(result, expected, rtol=1e-2, atol=2e-2):
            print(f"NUMERICAL MISMATCH:")
            print(f"Input wait_tensor_871 shape: {wait_tensor_871.shape}, dtype: {wait_tensor_871.dtype}")
            print(f"Input arg583_1 shape: {arg583_1.shape}, dtype: {arg583_1.dtype}")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            print(f"Expected (first few): {expected.flatten()[:10]}")
            print(f"Got (first few): {result.flatten()[:10]}")
            print(f"Max absolute difference: {torch.max(torch.abs(result - expected))}")
            print(f"Relative error: {torch.max(torch.abs((result - expected) / (expected + 1e-8)))}")
            
            # Additional debugging for embedding operations
            print(f"Input indices range: [{torch.min(arg583_1)}, {torch.max(arg583_1)}]")
            print(f"Expected embedding table shape would be: [128256, 4096]")
            return False

        print("Test passed successfully!")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        elif isinstance(e, RuntimeError) and "CUDA" in str(e):
            print(f"Test failed: CUDA error: {e}")
        else:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)