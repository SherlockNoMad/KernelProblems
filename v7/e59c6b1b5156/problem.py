# Fused region (lm_head): _to_copy -> _log_softmax -> nll_loss_forward -> div
# Instances: 1. Ops: 8, compute: 4, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_224: "bf16[8192, 128256][128256, 1]cuda:0",
        view_805: "i64[8192][1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1380} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 128256][1050673152, 128256, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_224, [1, 8192, 128256]);  mm_224 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        view_default: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default, [8192, 128256]);  _unsafe_view_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        _to_copy_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default, dtype = torch.float32);  view_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        _log_softmax_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._log_softmax.default(_to_copy_default, 1, False);  _to_copy_default = None
        nll_loss_forward_default = torch.ops.aten.nll_loss_forward.default(_log_softmax_default, view_805, None, 2, -100);  _log_softmax_default = view_805 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        getitem: "f32[][]cuda:0" = nll_loss_forward_default[0]

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        div_tensor: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(getitem, 8192.0);  getitem = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        getitem_1: "f32[][]cuda:0" = nll_loss_forward_default[1];  nll_loss_forward_default = None
        return (div_tensor, getitem_1)


def get_inputs():
    mm_224 = torch.randn([8192, 128256], dtype=torch.bfloat16, device='cuda')
    view_805 = torch.randint(0, 100, [8192], dtype=torch.int64, device='cuda')
    return [mm_224, view_805]

def get_init_inputs():
    return []
