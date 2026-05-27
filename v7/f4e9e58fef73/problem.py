# Fused region (layers.*.ffn_norm): _fused_rms_norm
# Instances: 32. Ops: 5, compute: 1, outputs: 3.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        add_recomputed: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_428: "bf16[4096][1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 23} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_recomputed, [4096], _unsafe_view_428, 1e-05);  add_recomputed = _unsafe_view_428 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 23} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 23} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096])

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 23} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096]);  getitem = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 23} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None
        return (view_default, view_default_1, getitem_1)


def get_inputs():
    add_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_428 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [add_recomputed, _unsafe_view_428]

def get_init_inputs():
    return []
