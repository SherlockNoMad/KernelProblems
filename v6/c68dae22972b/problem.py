# Fused region (layers.*.feed_forward.w1): silu -> mul
# Instances: 32. Ops: 5, compute: 2, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_221: "bf16[8192, 14336][14336, 1]cuda:0",
        mm_222: "bf16[8192, 14336][14336, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 873} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_221, [1, 8192, 14336]);  mm_221 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 873} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        silu_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.silu.default(_unsafe_view_default);  _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 873} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_222, [1, 8192, 14336]);  mm_222 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 873} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(silu_default, _unsafe_view_default_1);  silu_default = _unsafe_view_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 873} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.view.default(mul_tensor, [8192, 14336]);  mul_tensor = None
        return view_default


def get_inputs():
    mm_221 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    mm_222 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    return [mm_221, mm_222]

def get_init_inputs():
    return []
