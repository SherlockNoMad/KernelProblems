# Fused region (loss): _to_copy -> _log_softmax -> nll_loss_forward -> div -> ones_like -> div -> nll_loss_backward -> _log_softmax_backward_data -> _to_copy
# Instances: 1. Ops: 17, compute: 9, outputs: 1.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        arg584_1: "i64[1, 8192][8192, 1]cuda:0",
        mm_224: "bf16[8192, 128256][128256, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True} No stacktrace found for following nodes
        reshape_default: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(arg584_1, [8192]);  arg584_1 = None

        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[1, 8192, 128256][1050673152, 128256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [1, 8192, 128256]);  mm_224 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True} No stacktrace found for following nodes
        reshape_default_2: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.reshape.default(reshape_default_1, [8192, 128256]);  reshape_default_1 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        _to_copy_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._to_copy.default(reshape_default_2, dtype = torch.float32);  reshape_default_2 = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'decomposable', 'is_fusible': True} No stacktrace found for following nodes
        _log_softmax_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._log_softmax.default(_to_copy_default, 1, False);  _to_copy_default = None
        nll_loss_forward_default = torch.ops.aten.nll_loss_forward.default(_log_softmax_default, reshape_default, None, 2, -100)

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True} No stacktrace found for following nodes
        getitem: "f32[][]cuda:0" = nll_loss_forward_default[0]

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        div_tensor: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(getitem, 8192.0);  getitem = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE No stacktrace found for following nodes
        ones_like_default: "f32[][]cuda:0" = torch.ops.aten.ones_like.default(div_tensor, pin_memory = False, memory_format = torch.preserve_format);  div_tensor = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        div_tensor_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(ones_like_default, 8192.0);  ones_like_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True} No stacktrace found for following nodes
        getitem_1: "f32[][]cuda:0" = nll_loss_forward_default[1];  nll_loss_forward_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'decomposable', 'is_fusible': True} No stacktrace found for following nodes
        nll_loss_backward_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.nll_loss_backward.default(div_tensor_1, _log_softmax_default, reshape_default, None, 2, -100, getitem_1);  div_tensor_1 = reshape_default = getitem_1 = None
        _log_softmax_backward_data_default: "f32[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._log_softmax_backward_data.default(nll_loss_backward_default, _log_softmax_default, 1, torch.float32);  nll_loss_backward_default = _log_softmax_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        _to_copy_default_1: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten._to_copy.default(_log_softmax_backward_data_default, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  _log_softmax_backward_data_default = None

        # Annotation: {'module_fqn': 'loss', 'fusion_class': 'view', 'is_fusible': True} No stacktrace found for following nodes
        reshape_default_3: "bf16[1, 8192, 128256][1050673152, 128256, 1]cuda:0" = torch.ops.aten.reshape.default(_to_copy_default_1, [1, 8192, 128256]);  _to_copy_default_1 = None

        # Annotation: {'module_fqn': 'lm_head', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_4: "bf16[8192, 128256][128256, 1]cuda:0" = torch.ops.aten.reshape.default(reshape_default_3, [8192, 128256]);  reshape_default_3 = None
        t_default: "bf16[128256, 8192][1, 128256]cuda:0" = torch.ops.aten.t.default(reshape_default_4);  reshape_default_4 = None
        return t_default


def get_inputs():
    arg584_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    mm_224 = torch.randn([8192, 128256], dtype=torch.bfloat16, device='cuda')
    return [arg584_1, mm_224]

def get_init_inputs():
    return []
