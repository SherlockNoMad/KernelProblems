# Fused region (layers.*.attention.wo): add -> add
# Instances: 1. Ops: 4, compute: 2, outputs: 2.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        mm_220: "bf16[8192, 4096][4096, 1]cuda:0",
        add_61: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention.wo', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_220, [1, 8192, 4096])

        # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:51 in forward, code: h = x + self.attention(
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, reshape_default);  reshape_default = None

        # Annotation: {'module_fqn': 'layers.31.attention.wo', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_220, [1, 8192, 4096]);  mm_220 = None

        # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:51 in forward, code: h = x + self.attention(
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, reshape_default_1);  add_61 = reshape_default_1 = None
        return (add_tensor, add_tensor_1)


def get_inputs():
    mm_220 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_61 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [mm_220, add_61]

def get_init_inputs():
    return []
