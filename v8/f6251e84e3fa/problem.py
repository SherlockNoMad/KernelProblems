# Fused region (layers.*.attention_norm): add -> add -> _fused_rms_norm_backward -> add -> _to_copy -> embedding_dense_backward -> _to_copy
# Instances: 1. Ops: 13, compute: 7, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        _fused_rms_norm_recomputed,
        mm_670: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_672: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_674: "bf16[8192, 4096][4096, 1]cuda:0",
        embedding: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_432: "bf16[4096][1]cuda:0",
        add_220: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        arg583_1: "i64[1, 8192][8192, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1469} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_recomputed[1];  _fused_rms_norm_recomputed = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_670, [1, 8192, 4096]);  mm_670 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_672, [1, 8192, 4096]);  mm_672 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1469} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_674, [1, 8192, 4096]);  mm_674 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1469} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor_1, embedding, [4096], getitem, _unsafe_view_432, [True, True]);  add_tensor_1 = embedding = getitem = _unsafe_view_432 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]
        getitem_2: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1469} No stacktrace found for following nodes
        add_tensor_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_220, getitem_1);  add_220 = getitem_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.float32);  getitem_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/nn/modules/sparse.py:189 in forward, code: return F.embedding(
        embedding_dense_backward_default: "bf16[128256, 4096][4096, 1]cuda:0" = torch.ops.aten.embedding_dense_backward.default(add_tensor_2, arg583_1, 128256, -1, False);  add_tensor_2 = arg583_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1469} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default_1: "f32[128256, 4096][4096, 1]cuda:0" = torch.ops.aten._to_copy.default(embedding_dense_backward_default, dtype = torch.float32);  embedding_dense_backward_default = None
        return (_to_copy_default, _to_copy_default_1)


def get_inputs():
    _fused_rms_norm_recomputed = torch.randn([], dtype=torch.float32, device='cuda')
    mm_670 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_672 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_674 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    embedding = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_432 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    add_220 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    arg583_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    return [_fused_rms_norm_recomputed, mm_670, mm_672, mm_674, embedding, _unsafe_view_432, add_220, arg583_1]

def get_init_inputs():
    return []
