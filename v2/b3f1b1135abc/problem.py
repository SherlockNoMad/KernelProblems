# Fused region (layers.*.attention.inner_attention): _to_copy -> _to_copy
# Instances: 32. Ops: 10, compute: 2, outputs: 3.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        getitem_641: "bf16[1, 32, 8192, 128][33554432, 128, 4096, 1]cuda:0",
        getitem_642: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0",
        getitem_643: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_641, 1, 2);  getitem_641 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(transpose_int, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  transpose_int = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.reshape.default(_to_copy_default, [1, 8192, 32, 64, 2]);  _to_copy_default = None

        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int_1: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_642, 1, 2);  getitem_642 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(transpose_int_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  transpose_int_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_1: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.reshape.default(_to_copy_default_1, [1, 8192, 8, 64, 2]);  _to_copy_default_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int_2: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_643, 1, 2);  getitem_643 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:535 in forward, code: xv = xv.view(bs, seqlen, -1, self.head_dim)
        reshape_default_2: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(transpose_int_2, [1, 8192, 1024]);  transpose_int_2 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(reshape_default_2, [8192, 1024]);  reshape_default_2 = None
        t_default: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.t.default(reshape_default_3);  reshape_default_3 = None
        return (reshape_default, reshape_default_1, t_default)


def get_inputs():
    getitem_641 = torch.randn((33554432,), dtype=torch.bfloat16, device='cuda').as_strided([1, 32, 8192, 128], [33554432, 128, 4096, 1])
    getitem_642 = torch.randn((8388608,), dtype=torch.bfloat16, device='cuda').as_strided([1, 8, 8192, 128], [8388608, 128, 1024, 1])
    getitem_643 = torch.randn((8388608,), dtype=torch.bfloat16, device='cuda').as_strided([1, 8, 8192, 128], [8388608, 128, 1024, 1])
    return [getitem_641, getitem_642, getitem_643]

def get_init_inputs():
    return []
