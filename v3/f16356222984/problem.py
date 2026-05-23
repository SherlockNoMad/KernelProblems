# Fused region (layers.*.feed_forward.w2): _to_copy
# Instances: 32. Ops: 3, compute: 1, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_661: "bf16[4096, 14336][14336, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        t_default: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.t.default(mm_661);  mm_661 = None
        t_default_1: "bf16[4096, 14336][14336, 1]cuda:0" = torch.ops.aten.t.default(t_default);  t_default = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096, 14336][14336, 1]cuda:0" = torch.ops.aten._to_copy.default(t_default_1, dtype = torch.float32);  t_default_1 = None
        return _to_copy_default


def get_inputs():
    mm_661 = torch.randn([4096, 14336], dtype=torch.bfloat16, device='cuda')
    return [mm_661]

def get_init_inputs():
    return []
