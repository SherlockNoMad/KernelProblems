# Fused region (layers.*.feed_forward.w2): add
# Instances: 62. Ops: 2, compute: 1, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_216: "bf16[8192, 4096][4096, 1]cuda:0",
        add_60: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.30.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_216, [1, 8192, 4096]);  mm_216 = None

        # Annotation: {'module_fqn': 'layers.30', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: MUST_SAVE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:54 in forward, code: out = h + self.feed_forward(self.ffn_norm(h))
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_60, _unsafe_view_default);  add_60 = _unsafe_view_default = None
        return add_tensor


def get_inputs():
    mm_216 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_60 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [mm_216, add_60]

def get_init_inputs():
    return []
