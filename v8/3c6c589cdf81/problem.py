# Fused region (layers.*.attention.wo): add -> add -> _fused_rms_norm -> _fused_rms_norm_backward -> _to_copy
# Instances: 1. Ops: 12, compute: 5, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_220: "bf16[8192, 4096][4096, 1]cuda:0",
        add_61: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        mm_223: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_226: "bf16[8192, 4096][4096, 1]cuda:0",
        view_3253: "bf16[4096][1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.attention.wo', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_220, [1, 8192, 4096]);  mm_220 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 609} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:51 in forward, code: h = x + self.attention(
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, _unsafe_view_default);  add_61 = _unsafe_view_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_223, [1, 8192, 4096]);  mm_223 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 609} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:54 in forward, code: out = h + self.feed_forward(self.ffn_norm(h))
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_tensor, _unsafe_view_default_1);  add_tensor = _unsafe_view_default_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.view.default(mm_226, [1, 8192, 4096]);  mm_226 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 609} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor_1, [4096], view_3253, 1e-05)

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 609} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(view_default, add_tensor_1, [4096], getitem, view_3253, [True, True]);  view_default = add_tensor_1 = getitem = view_3253 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # autograd_backward: True # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_1, [8192, 4096]);  getitem_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 609} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_2: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 609} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.float32);  getitem_2 = None
        return (view_default_1, _to_copy_default)


def get_inputs():
    mm_220 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_61 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_223 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_226 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    view_3253 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [mm_220, add_61, mm_223, mm_226, view_3253]

def get_init_inputs():
    return []
