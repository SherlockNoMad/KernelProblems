# Fused region (layers.*.attention.wo): add -> _fused_rms_norm
# Instances: 32. Ops: 6, compute: 2, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_220: "bf16[8192, 4096][4096, 1]cuda:0",
        add_61: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        _unsafe_view_981: "bf16[4096][1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention.wo', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 604} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_220, [1, 8192, 4096]);  mm_220 = None

        # Annotation: {'module_fqn': 'layers.31', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 604} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:51 in forward, code: h = x + self.attention(
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, _unsafe_view_default);  add_61 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 604} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor, [4096], _unsafe_view_981, 1e-05);  add_tensor = _unsafe_view_981 = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 604} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 604} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 604} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem, [8192, 4096]);  getitem = None
        return (view_default, view_default_1)


def get_inputs():
    mm_220 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_61 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    _unsafe_view_981 = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [mm_220, add_61, _unsafe_view_981]

def get_init_inputs():
    return []
