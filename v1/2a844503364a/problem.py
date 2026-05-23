# Fused region (layers.*.attention_norm): _fused_rms_norm
# Instances: 32. Ops: 19, compute: 1, outputs: 5.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        getitem_1619: "bf16[8, 512][27264000, 1]cuda:0",
        add_62: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        getitem_1620: "bf16[8, 7340032][27264000, 1]cuda:0",
        getitem_1621: "bf16[8, 7340032][27264000, 1]cuda:0",
        getitem_1622: "bf16[8, 7340032][27264000, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1619, torch.bfloat16);  getitem_1619 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'decomposable', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_62, [4096], _unsafe_view_default, 1e-05);  add_62 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.31.ffn_norm', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        reshape_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(getitem, [8192, 4096]);  getitem = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype_1: "bf16[8, 7340032][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1620, torch.bfloat16);  getitem_1620 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_1: "bf16[8, 7340032][7340032, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_1, memory_format = torch.contiguous_format);  view_dtype_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_1: "bf16[14336, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_1, [14336, 4096]);  clone_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w1', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        t_default: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.t.default(_unsafe_view_default_1);  _unsafe_view_default_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype_2: "bf16[8, 7340032][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1621, torch.bfloat16);  getitem_1621 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_2: "bf16[8, 7340032][7340032, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_2, memory_format = torch.contiguous_format);  view_dtype_2 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_2: "bf16[14336, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_2, [14336, 4096]);  clone_default_2 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w3', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        t_default_1: "bf16[4096, 14336][1, 4096]cuda:0" = torch.ops.aten.t.default(_unsafe_view_default_2);  _unsafe_view_default_2 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_dtype_3: "bf16[8, 7340032][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1622, torch.bfloat16);  getitem_1622 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_3: "bf16[8, 7340032][7340032, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_3, memory_format = torch.contiguous_format);  view_dtype_3 = None

        # Annotation: {'module_fqn': 'layers.31.attention_norm', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_3: "bf16[4096, 14336][14336, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_3, [4096, 14336]);  clone_default_3 = None

        # Annotation: {'module_fqn': 'layers.31.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        t_default_2: "bf16[14336, 4096][1, 14336]cuda:0" = torch.ops.aten.t.default(_unsafe_view_default_3);  _unsafe_view_default_3 = None
        return (reshape_default, reshape_default_1, t_default, t_default_1, t_default_2)


def get_inputs():
    getitem_1619 = torch.randn((190848512,), dtype=torch.bfloat16, device='cuda').as_strided([8, 512], [27264000, 1])
    add_62 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    getitem_1620 = torch.randn((198188032,), dtype=torch.bfloat16, device='cuda').as_strided([8, 7340032], [27264000, 1])
    getitem_1621 = torch.randn((198188032,), dtype=torch.bfloat16, device='cuda').as_strided([8, 7340032], [27264000, 1])
    getitem_1622 = torch.randn((198188032,), dtype=torch.bfloat16, device='cuda').as_strided([8, 7340032], [27264000, 1])
    return [getitem_1619, add_62, getitem_1620, getitem_1621, getitem_1622]

def get_init_inputs():
    return []
