# Fused region (layers.*.feed_forward.w2): _fused_rms_norm -> add -> add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 31. Ops: 19, compute: 6, outputs: 5.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        getitem_818: "bf16[8, 512][27264000, 1]cuda:0",
        add_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        mm_656: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_658: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_660: "bf16[8192, 4096][4096, 1]cuda:0",
        add_215: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.1.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_818, torch.bfloat16);  getitem_818 = None

        # Annotation: {'module_fqn': 'layers.1.feed_forward.w2', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.1.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_1, [4096], _unsafe_view_default, 1e-05)

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096]);  getitem = None

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_819: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_656, [1, 8192, 4096]);  mm_656 = None

        # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_4: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_658, [1, 8192, 4096]);  mm_658 = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(reshape_default_3, reshape_default_4);  reshape_default_3 = reshape_default_4 = None

        # Annotation: {'module_fqn': 'layers.1.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_5: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_660, [1, 8192, 4096]);  mm_660 = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_5);  add_tensor = reshape_default_5 = None

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor_1, add_1, [4096], getitem_819, _unsafe_view_default, [True, True]);  add_tensor_1 = add_1 = getitem_819 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_820: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_215, getitem_820);  add_215 = getitem_820 = None

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_821: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # Annotation: {'module_fqn': 'layers.1.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_821, dtype = torch.float32);  getitem_821 = None
        return (reshape_default, reshape_default_1, reshape_default_2, add_tensor_2, _to_copy_default)


def get_inputs():
    getitem_818 = torch.randn((190848512,), dtype=torch.bfloat16, device='cuda').as_strided([8, 512], [27264000, 1])
    add_1 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_656 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_658 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_660 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_215 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [getitem_818, add_1, mm_656, mm_658, mm_660, add_215]

def get_init_inputs():
    return []
