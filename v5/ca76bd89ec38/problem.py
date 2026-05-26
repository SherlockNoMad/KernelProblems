# Fused region (layers.*.attention): index -> _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy -> _to_copy -> view_as_complex -> mul -> view_as_real -> _to_copy -> _to_copy -> view_as_complex -> _to_copy -> view_as_complex -> index -> mul -> view_as_real -> mul -> view_as_real -> _to_copy -> _to_copy
# Instances: 32. Ops: 46, compute: 22, outputs: 4.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        arg586_1: "i32[1, 8192][8192, 1]cuda:0",
        arg582_1: "c64[8192, 64][64, 1]cuda:0",
        mm_217: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_218: "bf16[8192, 1024][1024, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim]);  squeeze_dim = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.view.default(index_tensor, [1, 8192, 1, 64]);  index_tensor = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_217, [1, 8192, 4096])

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:533 in forward, code: xq = xq.view(bs, seqlen, -1, self.head_dim)
        view_default_1: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default, [1, 8192, -1, 128]);  _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_1, dtype = torch.float32);  view_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_2: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default, [1, 8192, 32, -1, 2]);  _to_copy_default = None
        view_as_complex_default: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_2);  view_default_2 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default, view_default);  view_as_complex_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor);  mul_tensor = None
        view_default_3: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default, [1, 8192, 32, 128]);  view_as_real_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_3, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int: "bf16[1, 32, 8192, 128][33554432, 128, 4096, 1]cuda:0" = torch.ops.aten.transpose.int(_to_copy_default_1, 1, 2);  _to_copy_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_1: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_218, [1, 8192, 1024])

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        view_default_4: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default_1, [1, 8192, -1, 128]);  _unsafe_view_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_2: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_4, dtype = torch.float32);  view_default_4 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_5: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_2, [1, 8192, 8, -1, 2]);  _to_copy_default_2 = None
        view_as_complex_default_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_5);  view_default_5 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default_1, view_default);  view_as_complex_default_1 = view_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default_1: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor_1);  mul_tensor_1 = None
        view_default_6: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default_1, [1, 8192, 8, 128]);  view_as_real_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_3: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_6, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_6 = None

        # Annotation: {'module_fqn': 'layers.31.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int_1: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0" = torch.ops.aten.transpose.int(_to_copy_default_3, 1, 2);  _to_copy_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_217, [1, 8192, 4096]);  mm_217 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_3: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_218, [1, 8192, 1024]);  mm_218 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:533 in forward, code: xq = xq.view(bs, seqlen, -1, self.head_dim)
        view_default_7: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default_2, [1, 8192, -1, 128]);  _unsafe_view_default_2 = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        view_default_8: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default_3, [1, 8192, -1, 128]);  _unsafe_view_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_4: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_7, dtype = torch.float32);  view_default_7 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_9: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_4, [1, 8192, 32, -1, 2]);  _to_copy_default_4 = None
        view_as_complex_default_2: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_9);  view_default_9 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_5: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_8, dtype = torch.float32);  view_default_8 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_10: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_5, [1, 8192, 8, -1, 2]);  _to_copy_default_5 = None
        view_as_complex_default_3: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_10);  view_default_10 = None
        squeeze_dim_1: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0);  arg586_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_1: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_1]);  arg582_1 = squeeze_dim_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_11: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.view.default(index_tensor_1, [1, 8192, 1, 64]);  index_tensor_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_2: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default_2, view_default_11);  view_as_complex_default_2 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default_2: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor_2);  mul_tensor_2 = None
        view_default_12: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default_2, [1, 8192, 32, 128]);  view_as_real_default_2 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_3: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default_3, view_default_11);  view_as_complex_default_3 = view_default_11 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default_3: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor_3);  mul_tensor_3 = None
        view_default_13: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default_3, [1, 8192, 8, 128]);  view_as_real_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_6: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_12, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_12 = None
        _to_copy_default_7: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_13, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_13 = None

        # Annotation: {'module_fqn': 'layers.31.attention.inner_attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:293 in forward, code: q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        transpose_int_2: "bf16[1, 32, 8192, 128][33554432, 128, 4096, 1]cuda:0" = torch.ops.aten.transpose.int(_to_copy_default_6, 1, 2);  _to_copy_default_6 = None
        transpose_int_3: "bf16[1, 8, 8192, 128][8388608, 128, 1024, 1]cuda:0" = torch.ops.aten.transpose.int(_to_copy_default_7, 1, 2);  _to_copy_default_7 = None
        return (transpose_int, transpose_int_1, transpose_int_2, transpose_int_3)


def get_inputs():
    arg586_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int32, device='cuda')
    arg582_1 = torch.randn([8192, 64], dtype=torch.complex64, device='cuda')
    mm_217 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_218 = torch.randn([8192, 1024], dtype=torch.bfloat16, device='cuda')
    return [arg586_1, arg582_1, mm_217, mm_218]

def get_init_inputs():
    return []
