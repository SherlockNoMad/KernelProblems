# Fused region (layers.*.feed_forward.w2): add -> split_with_sizes -> _fused_rms_norm
# Instances: 30. Ops: 40, compute: 3, outputs: 10.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        mm_209: "bf16[8192, 4096][4096, 1]cuda:0",
        add_58: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        wait_tensor_963: "bf16[218112000][1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.29.feed_forward.w2', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        _unsafe_view_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(mm_209, [1, 8192, 4096]);  mm_209 = None

        # Annotation: {'module_fqn': 'layers.29', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} recompute: MUST_SAVE File: /data/users/bahuang/torchtitan/torchtitan/models/llama3/model.py:54 in forward, code: out = h + self.feed_forward(self.ffn_norm(h))
        add_tensor: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_58, _unsafe_view_default);  add_58 = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_default: "bf16[8, 27264000][27264000, 1]cuda:0" = torch.ops.aten.view.default(wait_tensor_963, [8, -1]);  wait_tensor_963 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(view_default, [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032], 1);  view_default = None
        getitem: "bf16[8, 512][27264000, 1]cuda:0" = split_with_sizes_default[0]
        view_dtype: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem, torch.bfloat16);  getitem = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype, memory_format = torch.contiguous_format);  view_dtype = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_1: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 571} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(add_tensor, [4096], _unsafe_view_default_1, 1e-05);  add_tensor = _unsafe_view_default_1 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_1: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.30.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_1: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_1, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.30.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_1, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.30.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_1, [8192, 4096]);  getitem_1 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        getitem_2: "bf16[8, 2097152][27264000, 1]cuda:0" = split_with_sizes_default[1]
        view_dtype_1: "bf16[8, 2097152][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_2, torch.bfloat16);  getitem_2 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_1: "bf16[8, 2097152][2097152, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_1, memory_format = torch.contiguous_format);  view_dtype_1 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_2: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_1, [4096, 4096]);  clone_default_1 = None
        getitem_3: "bf16[8, 524288][27264000, 1]cuda:0" = split_with_sizes_default[2]
        view_dtype_2: "bf16[8, 524288][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_3, torch.bfloat16);  getitem_3 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_2: "bf16[8, 524288][524288, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_2, memory_format = torch.contiguous_format);  view_dtype_2 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_3: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_2, [1024, 4096]);  clone_default_2 = None
        getitem_4: "bf16[8, 524288][27264000, 1]cuda:0" = split_with_sizes_default[3]
        view_dtype_3: "bf16[8, 524288][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_4, torch.bfloat16);  getitem_4 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_3: "bf16[8, 524288][524288, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_3, memory_format = torch.contiguous_format);  view_dtype_3 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_4: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_3, [1024, 4096]);  clone_default_3 = None
        getitem_5: "bf16[8, 2097152][27264000, 1]cuda:0" = split_with_sizes_default[4]
        view_dtype_4: "bf16[8, 2097152][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_5, torch.bfloat16);  getitem_5 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_4: "bf16[8, 2097152][2097152, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_4, memory_format = torch.contiguous_format);  view_dtype_4 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_5: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_4, [4096, 4096]);  clone_default_4 = None
        getitem_6: "bf16[8, 512][27264000, 1]cuda:0" = split_with_sizes_default[5]
        view_dtype_5: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_6, torch.bfloat16);  getitem_6 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_5: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_5, memory_format = torch.contiguous_format);  view_dtype_5 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_6: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_5, [4096]);  clone_default_5 = None
        getitem_7: "bf16[8, 7340032][27264000, 1]cuda:0" = split_with_sizes_default[6]
        view_dtype_6: "bf16[8, 7340032][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_7, torch.bfloat16);  getitem_7 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_6: "bf16[8, 7340032][7340032, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_6, memory_format = torch.contiguous_format);  view_dtype_6 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_7: "bf16[14336, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_6, [14336, 4096]);  clone_default_6 = None
        getitem_8: "bf16[8, 7340032][27264000, 1]cuda:0" = split_with_sizes_default[7];  split_with_sizes_default = None
        view_dtype_7: "bf16[8, 7340032][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_8, torch.bfloat16);  getitem_8 = None

        # Annotation: {'module_fqn': 'layers.30.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 571} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_7: "bf16[8, 7340032][7340032, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_7, memory_format = torch.contiguous_format);  view_dtype_7 = None
        return (view_default_1, view_default_2, view_default_3, _unsafe_view_default_2, _unsafe_view_default_3, _unsafe_view_default_4, _unsafe_view_default_5, _unsafe_view_default_6, _unsafe_view_default_7, clone_default_7)


def get_inputs():
    mm_209 = torch.randn([8192, 4096], dtype=torch.bfloat16, device='cuda')
    add_58 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    wait_tensor_963 = torch.randn([218112000], dtype=torch.bfloat16, device='cuda')
    return [mm_209, add_58, wait_tensor_963]

def get_init_inputs():
    return []
