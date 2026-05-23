# Fused region (layers.*.attention_norm): _fused_rms_norm
# Instances: 32. Ops: 8, compute: 1, outputs: 3.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        getitem_1614: "bf16[8, 512][27264000, 1]cuda:0",
        add_61: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1614, torch.bfloat16);  getitem_1614 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_61, [4096], _unsafe_view_default, 1e-05);  add_61 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.31.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096]);  getitem = None
        return (reshape_default, reshape_default_1, reshape_default_2)


def get_inputs():
    getitem_1614 = torch.randn((190848512,), dtype=torch.bfloat16, device='cuda').as_strided([8, 512], [27264000, 1])
    add_61 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [getitem_1614, add_61]

def get_init_inputs():
    return []
