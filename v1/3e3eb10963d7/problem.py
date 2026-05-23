# Fused region (layers.*.attention): _to_copy
# Instances: 32. Ops: 5, compute: 1, outputs: 1.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        view_as_real_126: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_as_real_126, [1, 8192, 8, 128]);  view_as_real_126 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(reshape_default, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  reshape_default = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        reshape_default_1: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(_to_copy_default, [1, 8192, 1024]);  _to_copy_default = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(reshape_default_1, [8192, 1024]);  reshape_default_1 = None
        t_default: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.t.default(reshape_default_2);  reshape_default_2 = None
        return t_default


def get_inputs():
    view_as_real_126 = torch.randn([1, 8192, 8, 64, 2], dtype=torch.float32, device='cuda')
    return [view_as_real_126]

def get_init_inputs():
    return []
