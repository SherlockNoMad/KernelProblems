# Fused region (layers.*.feed_forward.w2): add -> split_with_sizes -> _fused_rms_norm -> _fused_rms_norm_backward -> _to_copy
# Instances: 1. Ops: 14, compute: 5, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_223: "bf16[8192, 4096][4096, 1]cuda:0",
        add_62_recomputed: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        mm_226: "bf16[8192, 4096][4096, 1]cuda:0",
        wait_tensor_970: "bf16[4096][1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_223, [1, 8192, 4096]);  mm_223 = None

        # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:54 in forward, code: out = h + self.feed_forward(self.ffn_norm(h))
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62_recomputed, reshape_default);  add_62_recomputed = reshape_default = None

        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_226, [1, 8192, 4096]);  mm_226 = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.view.default(wait_tensor_970, [8, -1]);  wait_tensor_970 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(view_default, [512], 1);  view_default = None
        getitem: "bf16[8, 512][512, 1]cuda:0" = split_with_sizes_default[0];  split_with_sizes_default = None
        view_dtype: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.view.dtype(getitem, torch.bfloat16);  getitem = None
        view_default_1: "bf16[4096][1]cuda:0" = torch.ops.aten.view.default(view_dtype, [4096]);  view_dtype = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor, [4096], view_default_1, 1e-05)

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(reshape_default_1, add_tensor, [4096], getitem_1, view_default_1, [True, True]);  reshape_default_1 = add_tensor = getitem_1 = view_default_1 = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]
        getitem_3: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.float32);  getitem_3 = None
        return (getitem_2, _to_copy_default)


def get_inputs():
    mm_223 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_62_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_226 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    wait_tensor_970 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [mm_223, add_62_recomputed, mm_226, wait_tensor_970]

def get_init_inputs():
    return []
