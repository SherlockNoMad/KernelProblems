# Fused region (layers.*.feed_forward.w2): mul -> mul -> silu_backward
# Instances: 32. Ops: 6, compute: 3, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_662: "bf16[8192, 14336][14336, 1]cuda:0",
        silu_recomputed: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0",
        _unsafe_view_5_recomputed: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0",
        _unsafe_view_4_recomputed: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1458} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.view.default(mm_662, [1, 8192, 14336]);  mm_662 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1458} File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_default, silu_recomputed);  silu_recomputed = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1458} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.view.default(mul_tensor, [8192, 14336]);  mul_tensor = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1458} File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_default, _unsafe_view_5_recomputed);  view_default = _unsafe_view_5_recomputed = None
        silu_backward_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.silu_backward.default(mul_tensor_1, _unsafe_view_4_recomputed);  mul_tensor_1 = _unsafe_view_4_recomputed = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1458} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.view.default(silu_backward_default, [8192, 14336]);  silu_backward_default = None
        return (view_default_1, view_default_2)


def get_inputs():
    mm_662 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    silu_recomputed = torch.randn([1, 8192, 14336], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_5_recomputed = torch.randn([1, 8192, 14336], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_4_recomputed = torch.randn([1, 8192, 14336], dtype=torch.bfloat16, device='cuda')
    return [mm_662, silu_recomputed, _unsafe_view_5_recomputed, _unsafe_view_4_recomputed]

def get_init_inputs():
    return []
