# Fused region (layers.*.feed_forward.w1): silu -> mul -> mul -> mul -> silu_backward
# Instances: 32. Ops: 13, compute: 5, outputs: 3.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        mm_4: "bf16[8192, 14336][14336, 1]cuda:0",
        mm_5: "bf16[8192, 14336][14336, 1]cuda:0",
        mm_662: "bf16[8192, 14336][14336, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 8192, 14336]);  mm_4 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        silu_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.silu.default(reshape_default)

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 8192, 14336]);  mm_5 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(silu_default, reshape_default_1)

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_tensor, [8192, 14336]);  mul_tensor = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.reshape.default(mm_662, [1, 8192, 14336]);  mm_662 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor_1: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(reshape_default_3, silu_default);  silu_default = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_4: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 14336]);  mul_tensor_1 = None
        t_default: "bf16[14336, 8192][1, 14336]cuda:0" = torch.ops.aten.t.default(reshape_default_4);  reshape_default_4 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/feed_forward.py:54 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        mul_tensor_2: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.mul.Tensor(reshape_default_3, reshape_default_1);  reshape_default_3 = reshape_default_1 = None
        silu_backward_default: "bf16[1, 8192, 14336][117440512, 14336, 1]cuda:0" = torch.ops.aten.silu_backward.default(mul_tensor_2, reshape_default);  mul_tensor_2 = reshape_default = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_5: "bf16[8192, 14336][14336, 1]cuda:0" = torch.ops.aten.reshape.default(silu_backward_default, [8192, 14336]);  silu_backward_default = None
        t_default_1: "bf16[14336, 8192][1, 14336]cuda:0" = torch.ops.aten.t.default(reshape_default_5);  reshape_default_5 = None
        return (reshape_default_2, t_default, t_default_1)


def get_inputs():
    mm_4 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    mm_5 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    mm_662 = torch.randn([8192, 14336], dtype=torch.bfloat16, device='cuda')
    return [mm_4, mm_5, mm_662]

def get_init_inputs():
    return []
