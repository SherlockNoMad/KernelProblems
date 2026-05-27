# Fused region (): ones_like -> div -> nll_loss_backward -> _log_softmax_backward_data -> _to_copy
# Instances: 1. Ops: 7, compute: 5, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        div: "f32[][]cuda:0",
        _log_softmax: "f32[8192, 128256][128256, 1]cuda:0",
        view_805: "i64[8192][1]cuda:0",
        getitem_419: "f32[][]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1380} recompute: PREFER_RECOMPUTE No stacktrace found for following nodes
        ones_like_default: "f32[][]cuda:0" = torch.ops.aten.ones_like.default(div, pin_memory = False, memory_format = torch.preserve_format);  div = None

        # autograd_backward: True # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        div_tensor: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(ones_like_default, 8192.0);  ones_like_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'loss', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        nll_loss_backward_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.nll_loss_backward.default(div_tensor, _log_softmax, view_805, None, 2, -100, getitem_419);  div_tensor = view_805 = getitem_419 = None
        _log_softmax_backward_data_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._log_softmax_backward_data.default(nll_loss_backward_default, _log_softmax, 1, torch.float32);  nll_loss_backward_default = _log_softmax = None

        # autograd_backward: True # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        _to_copy_default: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._to_copy.default(_log_softmax_backward_data_default, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  _log_softmax_backward_data_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1380} No stacktrace found for following nodes
        view_default: "bf16[1, 8192, 128256][1050673152, 128256, 1]cuda:0" = torch.ops.aten.view.default(_to_copy_default, [1, 8192, 128256]);  _to_copy_default = None

        # autograd_backward: True # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 1380} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.view.default(view_default, [8192, 128256]);  view_default = None
        return view_default_1


def get_inputs():
    div = torch.randn([], dtype=torch.float32, device='cuda')
    _log_softmax = torch.randn([8192, 128256], dtype=torch.float32, device='cuda')
    view_805 = torch.randint(0, 100, [8192], dtype=torch.int64, device='cuda')
    getitem_419 = torch.randn([], dtype=torch.float32, device='cuda')
    return [div, _log_softmax, view_805, getitem_419]

def get_init_inputs():
    return []
