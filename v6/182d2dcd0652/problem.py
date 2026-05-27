# Fused region (layers.*.feed_forward.w1): silu -> mul -> mul -> mul -> silu_backward
# Instances: 32. Ops: 11, compute: 5, outputs: 3.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_4: "bf16[8192, 14336][14336, 1]cuda:0",
        mm_5: "bf16[8192, 14336][14336, 1]cuda:0",
        mm_662: "bf16[8192, 14336][14336, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 20} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_4, [1, 8192, 14336]);  mm_4 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 20} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        silu_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.silu.default(_unsafe_view_default)

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 20} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_5, [1, 8192, 14336]);  mm_5 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 20} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(silu_default, _unsafe_view_default_1)

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 20} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.view.default(mul_tensor, [8192, 14336]);  mul_tensor = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 20} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.view.default(mm_662, [1, 8192, 14336]);  mm_662 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 20} File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_default_1, silu_default);  silu_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 20} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.view.default(mul_tensor_1, [8192, 14336]);  mul_tensor_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 20} File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor_2: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_default_1, _unsafe_view_default_1);  view_default_1 = _unsafe_view_default_1 = None
        silu_backward_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.silu_backward.default(mul_tensor_2, _unsafe_view_default);  mul_tensor_2 = _unsafe_view_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 20} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.view.default(silu_backward_default, [8192, 14336]);  silu_backward_default = None
        return (view_default, view_default_2, view_default_3)


def get_inputs():
    mm_4 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    mm_5 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    mm_662 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    return [mm_4, mm_5, mm_662]

def get_init_inputs():
    return []
