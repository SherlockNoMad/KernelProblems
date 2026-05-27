# Fused region (layers.*.attention.wo): add -> _fused_rms_norm -> add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 31. Ops: 15, compute: 6, outputs: 4.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_3: "bf16[8192, 4096][4096, 1]cuda:0",
        embedding: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_428: "bf16[4096][1]cuda:0",
        mm_664: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_666: "bf16[8192, 4096][4096, 1]cuda:0",
        add_218: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.wo', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_3, [1, 8192, 4096]);  mm_3 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:51 in forward, code: h = x + self.attention(
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, _unsafe_view_default);  embedding = _unsafe_view_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor, [4096], _unsafe_view_428, 1e-05)

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096])

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096]);  getitem = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_664, [1, 8192, 4096]);  mm_664 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_666, [1, 8192, 4096]);  mm_666 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 13} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_default_2, view_default_3);  view_default_2 = view_default_3 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 13} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor_1, add_tensor, [4096], getitem_1, _unsafe_view_428, [True, True]);  add_tensor_1 = add_tensor = getitem_1 = _unsafe_view_428 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 13} No stacktrace found for following nodes
        add_tensor_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_218, getitem_2);  add_218 = getitem_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 13} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_3: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 13} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.float32);  getitem_3 = None
        return (view_default, view_default_1, add_tensor_2, _to_copy_default)


def get_inputs():
    mm_3 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    embedding = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_428 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    mm_664 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_666 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_218 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [mm_3, embedding, _unsafe_view_428, mm_664, mm_666, add_218]

def get_init_inputs():
    return []
