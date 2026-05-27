# Fused region (norm): _fused_rms_norm -> _fused_rms_norm_backward
# Instances: 1. Ops: 5, compute: 2, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        add_63_recomputed: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        view_3253: "bf16[4096][1]cuda:0",
        view_808: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1365} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_63_recomputed, [4096], view_3253, 1e-05)

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1365} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1365} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(view_808, add_63_recomputed, [4096], getitem, view_3253, [True, True]);  view_808 = add_63_recomputed = getitem = view_3253 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1365} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]
        getitem_2: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None
        return (getitem_1, getitem_2)


def get_inputs():
    add_63_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    view_3253 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    view_808 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [add_63_recomputed, view_3253, view_808]

def get_init_inputs():
    return []
