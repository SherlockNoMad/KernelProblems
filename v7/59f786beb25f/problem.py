# Fused region (layers.*.attention_norm): _fused_rms_norm
# Instances: 32. Ops: 3, compute: 1, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        embedding: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_432: "bf16[4096][1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 12} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(embedding, [4096], _unsafe_view_432, 1e-05);  embedding = _unsafe_view_432 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 12} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]
        getitem_1: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None
        return (getitem, getitem_1)


def get_inputs():
    embedding = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_432 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [embedding, _unsafe_view_432]

def get_init_inputs():
    return []
