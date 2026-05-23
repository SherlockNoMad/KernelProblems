# Fused region (layers.*.attention.qkv_linear.wk): _to_copy -> _to_copy
# Instances: 32. Ops: 8, compute: 2, outputs: 2.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        mm_218: "bf16[8192, 1024][1024, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [1, 8192, 1024])

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        reshape_default_1: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(reshape_default, [1, 8192, -1, 128]);  reshape_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(reshape_default_1, dtype = torch.float32);  reshape_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_2: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.reshape.default(_to_copy_default, [1, 8192, 8, -1, 2]);  _to_copy_default = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [1, 8192, 1024]);  mm_218 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        reshape_default_4: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(reshape_default_3, [1, 8192, -1, 128]);  reshape_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(reshape_default_4, dtype = torch.float32);  reshape_default_4 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_5: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.reshape.default(_to_copy_default_1, [1, 8192, 8, -1, 2]);  _to_copy_default_1 = None
        return (reshape_default_2, reshape_default_5)


def get_inputs():
    mm_218 = torch.randn([8192, 1024], dtype=torch.bfloat16, device='cuda')
    return [mm_218]

def get_init_inputs():
    return []
