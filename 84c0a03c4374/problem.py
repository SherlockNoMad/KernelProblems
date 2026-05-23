# Fused region (layers.*.feed_forward.w2): _fused_rms_norm -> add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 1. Ops: 16, compute: 5, outputs: 4.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        getitem_1624: "bf16[8, 512][27264000, 1]cuda:0",
        add_62_recomputed: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        mm_230: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_232: "bf16[8192, 4096][4096, 1]cuda:0",
        getitem_420: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1624, torch.bfloat16);  getitem_1624 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_62_recomputed, [4096], _unsafe_view_default, 1e-05)

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096]);  getitem = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1625: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_230, [1, 8192, 4096]);  mm_230 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_232, [1, 8192, 4096]);  mm_232 = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(reshape_default_2, reshape_default_3);  reshape_default_2 = reshape_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor, add_62_recomputed, [4096], getitem_1625, _unsafe_view_default, [True, True]);  add_tensor = add_62_recomputed = getitem_1625 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1626: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_420, getitem_1626);  getitem_420 = getitem_1626 = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_421: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_421, dtype = torch.float32);  getitem_421 = None
        return (reshape_default, reshape_default_1, add_tensor_1, _to_copy_default)


def get_inputs():
    getitem_1624 = torch.randn((190848512,), dtype=torch.bfloat16, device='cuda').as_strided([8, 512], [27264000, 1])
    add_62_recomputed = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_230 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_232 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    getitem_420 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [getitem_1624, add_62_recomputed, mm_230, mm_232, getitem_420]

def get_init_inputs():
    return []
