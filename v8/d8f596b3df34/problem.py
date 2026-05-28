# Fused region (layers.*.feed_forward.w3): add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 32. Ops: 9, compute: 4, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_664: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_666: "bf16[8192, 4096][4096, 1]cuda:0",
        add_recomputed: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        getitem_12_recomputed: "f32[1, 8192, 1][8192, 1, 1]cuda:0",
        _unsafe_view_428: "bf16[4096][1]cuda:0",
        add_218: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_664, [1, 8192, 4096]);  mm_664 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_666, [1, 8192, 4096]);  mm_666 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1461} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor, add_recomputed, [4096], getitem_12_recomputed, _unsafe_view_428, [True, True]);  add_tensor = add_recomputed = getitem_12_recomputed = _unsafe_view_428 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1461} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_218, getitem);  add_218 = getitem = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.wo', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(add_tensor_1, [8192, 4096]);  add_tensor_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1461} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_1, dtype = torch.float32);  getitem_1 = None
        return (view_default_2, _to_copy_default)


def get_inputs():
    mm_664 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_666 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    getitem_12_recomputed = torch.randn([1, 8192, 1], dtype=torch.float32, device='cuda')
    _unsafe_view_428 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    add_218 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [mm_664, mm_666, add_recomputed, getitem_12_recomputed, _unsafe_view_428, add_218]

def get_init_inputs():
    return []
