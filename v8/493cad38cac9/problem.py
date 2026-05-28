# Fused region (layers.*.attention.qkv_linear.wq): _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy
# Instances: 31. Ops: 9, compute: 5, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_217: "bf16[8192, 4096][4096, 1]cuda:0",
        view_788: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_217, [1, 8192, 4096]);  mm_217 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:533 in forward, code: xq = xq.view(bs, seqlen, -1, self.head_dim)
        view_default: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default, [1, 8192, -1, 128]);  _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default, dtype = torch.float32);  view_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_1: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default, [1, 8192, 32, -1, 2]);  _to_copy_default = None
        view_as_complex_default: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_1);  view_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default, view_788);  view_as_complex_default = view_788 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor);  mul_tensor = None
        view_default_2: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default, [1, 8192, 32, 128]);  view_as_real_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 592} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_2, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_2 = None
        return _to_copy_default_1


def get_inputs():
    mm_217 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    view_788 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device='cuda')
    return [mm_217, view_788]

def get_init_inputs():
    return []
