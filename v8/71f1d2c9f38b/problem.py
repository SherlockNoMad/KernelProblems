# Fused region (loss): _to_copy -> _log_softmax -> nll_loss_forward -> div
# Instances: 1. Ops: 9, compute: 4, outputs: 2.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        arg584_1: "i64[1, 8192][8192, 1]cuda:0",
        mm_224: "bf16[8192, 128256][128256, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        view_default: "i64[8192][1]cuda:0" = torch.ops.aten.view.default(arg584_1, [8192]);  arg584_1 = None

        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 608} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 128256][1050673152, 128256, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_224, [1, 8192, 128256]);  mm_224 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        view_default_1: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.view.default(_unsafe_view_default, [8192, 128256]);  _unsafe_view_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        _to_copy_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._to_copy.default(view_default_1, dtype = torch.float32);  view_default_1 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        _log_softmax_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._log_softmax.default(_to_copy_default, 1, False);  _to_copy_default = None
        nll_loss_forward_default = torch.ops.aten.nll_loss_forward.default(_log_softmax_default, view_default, None, 2, -100);  _log_softmax_default = view_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        getitem: "f32[][]cuda:0" = nll_loss_forward_default[0]

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        div_tensor: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(getitem, 8192.0);  getitem = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 608} No stacktrace found for following nodes
        getitem_1: "f32[][]cuda:0" = nll_loss_forward_default[1];  nll_loss_forward_default = None
        return (div_tensor, getitem_1)


def get_inputs():
    arg584_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    mm_224 = torch.randn([8192, 128256], dtype=torch.bfloat16, device='cuda')
    return [arg584_1, mm_224]

def get_init_inputs():
    return []
