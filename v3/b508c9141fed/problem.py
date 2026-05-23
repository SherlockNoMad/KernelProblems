# Fused region (layers.*.attention): _to_copy
# Instances: 64. Ops: 3, compute: 1, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        view_as_real_1_recomputed: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_as_real_1_recomputed, [1, 8192, 8, 128]);  view_as_real_1_recomputed = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(reshape_default, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  reshape_default = None

        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0" = torch.ops.aten.transpose.int(_to_copy_default, 1, 2);  _to_copy_default = None
        return transpose_int


def get_inputs():
    view_as_real_1_recomputed = torch.randn([1, 8192, 8, 64, 2], dtype=torch.float32, device='cuda')
    return [view_as_real_1_recomputed]

def get_init_inputs():
    return []
