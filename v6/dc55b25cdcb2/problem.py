# Fused region (layers.*.attention): _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy
# Instances: 32. Ops: 11, compute: 5, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        _conj_63: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0",
        getitem_641: "bf16[1, 32, 8192, 128][33554432, 128, 4096, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        clone_default: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.clone.default(_conj_63);  _conj_63 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_641, 1, 2);  getitem_641 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(transpose_int, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  transpose_int = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default, [1, 8192, 32, 64, 2]);  _to_copy_default = None
        view_as_complex_default: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default);  view_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default, clone_default);  view_as_complex_default = clone_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor);  mul_tensor = None
        view_default_1: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default, [1, 8192, 32, 128]);  view_as_real_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:533 in forward, code: xq = xq.view(bs, seqlen, -1, self.head_dim)
        view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_1, [1, 8192, 4096]);  _to_copy_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 10} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(view_default_2, [8192, 4096]);  view_default_2 = None
        return view_default_3


def get_inputs():
    _conj_63 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device='cuda')
    getitem_641 = torch.randn((33554432,), dtype=torch.bfloat16, device='cuda').as_strided([1, 32, 8192, 128], [33554432, 128, 4096, 1])
    return [_conj_63, getitem_641]

def get_init_inputs():
    return []
