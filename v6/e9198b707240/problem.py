# Fused region (layers.*.ffn_norm): _fused_rms_norm -> add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 1. Ops: 13, compute: 5, outputs: 4.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        add_62_recomputed: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_986: "bf16[4096][1]cuda:0",
        mm_230: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_232: "bf16[8192, 4096][4096, 1]cuda:0",
        getitem_420: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 846} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_62_recomputed, [4096], _unsafe_view_986, 1e-05)

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096])

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096]);  getitem = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_230, [1, 8192, 4096]);  mm_230 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_232, [1, 8192, 4096]);  mm_232 = None

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 846} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(view_default_2, view_default_3);  view_default_2 = view_default_3 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 846} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor, add_62_recomputed, [4096], getitem_1, _unsafe_view_986, [True, True]);  add_tensor = add_62_recomputed = getitem_1 = _unsafe_view_986 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 846} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_420, getitem_2);  getitem_420 = getitem_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 846} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_421: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 846} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_421, dtype = torch.float32);  getitem_421 = None
        return (view_default, view_default_1, add_tensor_1, _to_copy_default)


def get_inputs():
    add_62_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_986 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    mm_230 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_232 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    getitem_420 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [add_62_recomputed, _unsafe_view_986, mm_230, mm_232, getitem_420]

def get_init_inputs():
    return []
