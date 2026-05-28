# Fused region (layers.*.attention_norm): add -> add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 31. Ops: 12, compute: 5, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        _fused_rms_norm_2_recomputed,
        mm_656: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_658: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_660: "bf16[8192, 4096][4096, 1]cuda:0",
        add_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_450: "bf16[4096][1]cuda:0",
        add_215: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_2_recomputed[1];  _fused_rms_norm_2_recomputed = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_656, [1, 8192, 4096]);  mm_656 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_658, [1, 8192, 4096]);  mm_658 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1451} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_660, [1, 8192, 4096]);  mm_660 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1451} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor_1, add_1, [4096], getitem, _unsafe_view_450, [True, True]);  add_tensor_1 = add_1 = getitem = _unsafe_view_450 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1451} No stacktrace found for following nodes
        add_tensor_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_215, getitem_1);  add_215 = getitem_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(add_tensor_2, [8192, 4096]);  add_tensor_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_2: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1451} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.float32);  getitem_2 = None
        return (view_default_3, _to_copy_default)


def get_inputs():
    _fused_rms_norm_2_recomputed = torch.randn([], dtype=torch.float32, device='cuda')
    mm_656 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_658 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_660 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_1 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_450 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    add_215 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [_fused_rms_norm_2_recomputed, mm_656, mm_658, mm_660, add_1, _unsafe_view_450, add_215]

def get_init_inputs():
    return []
