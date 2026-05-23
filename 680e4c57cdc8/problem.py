# Fused region (layers.*.feed_forward.w2): _fused_rms_norm -> add -> add -> _fused_rms_norm_backward -> add -> _to_copy -> embedding_dense_backward -> _to_copy
# Instances: 1. Ops: 21, compute: 8, outputs: 5.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        getitem_791: "bf16[8, 512][27264000, 1]cuda:0",
        embedding: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        mm_670: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_672: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_674: "bf16[8192, 4096][4096, 1]cuda:0",
        add_220: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        arg583_1: "i64[1, 8192][8192, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_791, torch.bfloat16);  getitem_791 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(embedding, [4096], _unsafe_view_default, 1e-05)

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096]);  getitem = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_792: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_670, [1, 8192, 4096]);  mm_670 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_4: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_672, [1, 8192, 4096]);  mm_672 = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(reshape_default_3, reshape_default_4);  reshape_default_3 = reshape_default_4 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_5: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_674, [1, 8192, 4096]);  mm_674 = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_5);  add_tensor = reshape_default_5 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor_1, embedding, [4096], getitem_792, _unsafe_view_default, [True, True]);  add_tensor_1 = embedding = getitem_792 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_793: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]
        getitem_794: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_220, getitem_793);  add_220 = getitem_793 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_794, dtype = torch.float32);  getitem_794 = None

        # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/sparse.py:189 in forward, code: return F.embedding(
        embedding_dense_backward_default: "bf16[128256, 4096][4096, 1]cuda:0" = torch.ops.aten.embedding_dense_backward.default(add_tensor_2, arg583_1, 128256, -1, False);  add_tensor_2 = arg583_1 = None

        # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default_1: "f32[128256, 4096][4096, 1]cuda:0" = torch.ops.aten._to_copy.default(embedding_dense_backward_default, dtype = torch.float32);  embedding_dense_backward_default = None
        return (reshape_default, reshape_default_1, reshape_default_2, _to_copy_default, _to_copy_default_1)


def get_inputs():
    getitem_791 = torch.randn((190848512,), dtype=torch.bfloat16, device='cuda').as_strided([8, 512], [27264000, 1])
    embedding = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_670 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_672 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_674 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_220 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    arg583_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    return [getitem_791, embedding, mm_670, mm_672, mm_674, add_220, arg583_1]

def get_init_inputs():
    return []
