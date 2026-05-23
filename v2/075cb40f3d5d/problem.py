# Fused region (layers.*.attention): index -> mul -> mul
# Instances: 31. Ops: 5, compute: 3, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        arg586_1: "i32[1, 8192][8192, 1]cuda:0",
        arg582_1: "c64[8192, 64][64, 1]cuda:0",
        view_as_complex_60: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_61: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0);  arg586_1 = None

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim]);  arg582_1 = squeeze_dim = None

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor, [1, 8192, 1, 64]);  index_tensor = None

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_60, reshape_default);  view_as_complex_60 = None
        mul_tensor_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_61, reshape_default);  view_as_complex_61 = reshape_default = None
        return (mul_tensor, mul_tensor_1)


def get_inputs():
    arg586_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int32, device='cuda')
    arg582_1 = torch.randn([8192, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_60 = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_61 = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    return [arg586_1, arg582_1, view_as_complex_60, view_as_complex_61]

def get_init_inputs():
    return []
