# Fused region (layers.*.attention): _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy -> _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy
# Instances: 32. Ops: 25, compute: 10, outputs: 3.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        _conj_62: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0",
        _conj_63: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0",
        getitem_641: "bf16[1, 32, 8192, 128][33554432, 128, 4096, 1]cuda:0",
        getitem_642: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0",
        getitem_643: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        clone_default: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.clone.default(_conj_62);  _conj_62 = None
        clone_default_1: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.clone.default(_conj_63);  _conj_63 = None

        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_641, 1, 2);  getitem_641 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(transpose_int, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  transpose_int = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default, [1, 8192, 32, 64, 2]);  _to_copy_default = None
        view_as_complex_default: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default);  view_default = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default, clone_default_1);  view_as_complex_default = clone_default_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor);  mul_tensor = None
        view_default_1: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default, [1, 8192, 32, 128]);  view_as_real_default = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:533 in forward, code: xq = xq.view(bs, seqlen, -1, self.head_dim)
        view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_1, [1, 8192, 4096]);  _to_copy_default_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(view_default_2, [8192, 4096]);  view_default_2 = None

        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int_1: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_642, 1, 2);  getitem_642 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_2: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(transpose_int_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  transpose_int_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_4: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_2, [1, 8192, 8, 64, 2]);  _to_copy_default_2 = None
        view_as_complex_default_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_4);  view_default_4 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default_1, clone_default);  view_as_complex_default_1 = clone_default = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default_1: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor_1);  mul_tensor_1 = None
        view_default_5: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default_1, [1, 8192, 8, 128]);  view_as_real_default_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_3: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_5, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_5 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        view_default_6: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_3, [1, 8192, 1024]);  _to_copy_default_3 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.view.default(view_default_6, [8192, 1024]);  view_default_6 = None

        # Annotation: {'module_fqn': 'layers.0.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int_2: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.transpose.int(getitem_643, 1, 2);  getitem_643 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:535 in forward, code: xv = xv.view(bs, seqlen, -1, self.head_dim)
        view_default_8: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.view.default(transpose_int_2, [1, 8192, 1024]);  transpose_int_2 = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.view.default(view_default_8, [8192, 1024]);  view_default_8 = None
        return (view_default_3, view_default_7, view_default_9)


def get_inputs():
    _conj_62 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device='cuda')
    _conj_63 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device='cuda')
    getitem_641 = torch.randn((33554432,), dtype=torch.bfloat16, device='cuda').as_strided([1, 32, 8192, 128], [33554432, 128, 4096, 1])
    getitem_642 = torch.randn((8388608,), dtype=torch.bfloat16, device='cuda').as_strided([1, 8, 8192, 128], [8388608, 128, 1024, 1])
    getitem_643 = torch.randn((8388608,), dtype=torch.bfloat16, device='cuda').as_strided([1, 8, 8192, 128], [8388608, 128, 1024, 1])
    return [_conj_62, _conj_63, getitem_641, getitem_642, getitem_643]

def get_init_inputs():
    return []
