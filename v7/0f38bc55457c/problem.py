# Fused region (layers.*.attention.qkv_linear.wq): _to_copy -> view_as_complex -> _to_copy -> view_as_complex -> index -> mul -> view_as_real -> mul -> view_as_real -> _to_copy
# Instances: 32. Ops: 20, compute: 10, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_1: "bf16[8192, 1024][1024, 1]cuda:0",
        arg586_1: "i32[1, 8192][8192, 1]cuda:0",
        arg582_1: "c64[8192, 64][64, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm, [1, 8192, 4096]);  mm = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_1: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_1, [1, 8192, 1024]);  mm_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:533 in forward, code: xq = xq.view(bs, seqlen, -1, self.head_dim)
        view_default: "bf16[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default, [1, 8192, -1, 128]);  _unsafe_view_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:534 in forward, code: xk = xk.view(bs, seqlen, -1, self.head_dim)
        view_default_1: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default_1, [1, 8192, -1, 128]);  _unsafe_view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default, dtype = torch.float32);  view_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_2: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default, [1, 8192, 32, -1, 2]);  _to_copy_default = None
        view_as_complex_default: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_2);  view_default_2 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_1: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_1, dtype = torch.float32);  view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_3: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default_1, [1, 8192, 8, -1, 2]);  _to_copy_default_1 = None
        view_as_complex_default_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.view_as_complex.default(view_default_3);  view_default_3 = None
        squeeze_dim: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0);  arg586_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim]);  arg582_1 = squeeze_dim = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_default_4: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.view.default(index_tensor, [1, 8192, 1, 64]);  index_tensor = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default, view_default_4);  view_as_complex_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default: "f32[1, 8192, 32, 64, 2][33554432, 4096, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor);  mul_tensor = None
        view_default_5: "f32[1, 8192, 32, 128][33554432, 4096, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default, [1, 8192, 32, 128]);  view_as_real_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_default_1, view_default_4);  view_as_complex_default_1 = view_default_4 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        view_as_real_default_1: "f32[1, 8192, 8, 64, 2][8388608, 1024, 128, 2, 1]cuda:0" = torch.ops.aten.view_as_real.default(mul_tensor_1);  mul_tensor_1 = None
        view_default_6: "f32[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten.view.default(view_as_real_default_1, [1, 8192, 8, 128]);  view_as_real_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 63} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        _to_copy_default_2: "bf16[1, 8192, 8, 128][8388608, 1024, 128, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_6, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  view_default_6 = None
        return (view_default_5, _to_copy_default_2)


def get_inputs():
    mm = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_1 = torch.randn([8192, 1024], dtype=torch.bfloat16, device='cuda')
    arg586_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int32, device='cuda')
    arg582_1 = torch.randn([8192, 64], dtype=torch.complex64, device='cuda')
    return [mm, mm_1, arg586_1, arg582_1]

def get_init_inputs():
    return []
