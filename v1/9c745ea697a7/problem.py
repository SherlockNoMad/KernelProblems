# Fused region (layers.*.attention.wo): add -> _fused_rms_norm -> add -> _fused_rms_norm_backward -> add -> _to_copy
# Instances: 31. Ops: 18, compute: 6, outputs: 4.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        mm_3: "bf16[8192, 4096][4096, 1]cuda:0",
        embedding: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        getitem_787: "bf16[8, 512][27264000, 1]cuda:0",
        mm_664: "bf16[8192, 4096][4096, 1]cuda:0",
        mm_666: "bf16[8192, 4096][4096, 1]cuda:0",
        add_218: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.attention.wo', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 8192, 4096]);  mm_3 = None

        # Annotation: {'module_fqn': 'layers.0', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:51 in forward, code: h = x + self.attention(
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, reshape_default);  embedding = reshape_default = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_787, torch.bfloat16);  getitem_787 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor, [4096], _unsafe_view_default, 1e-05)

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0]

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096]);  getitem = None

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_788: "f32[1, 8192, 1][8192, 1, 1]cuda:0" = _fused_rms_norm_default[1];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_3: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_664, [1, 8192, 4096]);  mm_664 = None

        # Annotation: {'module_fqn': 'layers.0.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_4: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_666, [1, 8192, 4096]);  mm_666 = None

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(reshape_default_3, reshape_default_4);  reshape_default_3 = reshape_default_4 = None

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_backward_default = torch.ops.aten._fused_rms_norm_backward.default(add_tensor_1, add_tensor, [4096], getitem_788, _unsafe_view_default, [True, True]);  add_tensor_1 = add_tensor = getitem_788 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_789: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_backward_default[0]

        # Annotation: {'fusion_class': 'pointwise', 'is_fusible': True} No stacktrace found for following nodes
        add_tensor_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_218, getitem_789);  add_218 = getitem_789 = None

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_790: "bf16[4096][1]cuda:0" = _fused_rms_norm_backward_default[1];  _fused_rms_norm_backward_default = None

        # Annotation: {'module_fqn': 'layers.0.ffn_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[4096][1]cuda:0" = torch.ops.aten._to_copy.default(getitem_790, dtype = torch.float32);  getitem_790 = None
        return (reshape_default_1, reshape_default_2, add_tensor_2, _to_copy_default)


def get_inputs():
    mm_3 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    embedding = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    getitem_787 = torch.randn((190848512,), dtype=torch.bfloat16, device='cuda').as_strided([8, 512], [27264000, 1])
    mm_664 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    mm_666 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_218 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    return [mm_3, embedding, getitem_787, mm_664, mm_666, add_218]

def get_init_inputs():
    return []
