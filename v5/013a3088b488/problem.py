# Fused region (layers.*.feed_forward.w2): add -> split_with_sizes -> _fused_rms_norm
# Instances: 1. Ops: 16, compute: 3, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_223: "bf16[8192, 4096][4096, 1]cuda:0",
        add_62: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        wait_tensor_969: "bf16[525340672][1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_223, [1, 8192, 4096]);  mm_223 = None

        # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:54 in forward, code: out = h + self.feed_forward(self.ffn_norm(h))
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, _unsafe_view_default);  add_62 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_default: "bf16[8, 65667584][65667584, 1]cuda:0" = torch.ops.aten.view.default(wait_tensor_969, [8, -1]);  wait_tensor_969 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(view_default, [512, 65667072], 1);  view_default = None
        getitem: "bf16[8, 512][65667584, 1]cuda:0" = split_with_sizes_default[0]
        view_dtype: "bf16[8, 512][65667584, 1]cuda:0" = torch.ops.aten.view.dtype(getitem, torch.bfloat16);  getitem = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_1: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor, [4096], _unsafe_view_default_1, 1e-05);  add_tensor = _unsafe_view_default_1 = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_1, [8192, 4096]);  getitem_1 = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        getitem_2: "bf16[8, 65667072][65667584, 1]cuda:0" = split_with_sizes_default[1];  split_with_sizes_default = None
        view_dtype_1: "bf16[8, 65667072][65667584, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_2, torch.bfloat16);  getitem_2 = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_1: "bf16[8, 65667072][65667072, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_1, memory_format = torch.contiguous_format);  view_dtype_1 = None

        # Annotation: {'module_fqn': 'norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_2: "bf16[128256, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_1, [128256, 4096]);  clone_default_1 = None

        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        t_default: "bf16[4096, 128256][1, 4096]cuda:0" = torch.ops.aten.t.default(_unsafe_view_default_2);  _unsafe_view_default_2 = None
        return (view_default_1, t_default)


def get_inputs():
    mm_223 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_62 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    wait_tensor_969 = torch.randn([525340672], dtype=torch.bfloat16, device='cuda')
    return [mm_223, add_62, wait_tensor_969]

def get_init_inputs():
    return []
