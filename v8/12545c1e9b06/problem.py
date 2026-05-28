# Fused region (tok_embeddings): split_with_sizes -> embedding -> split_with_sizes -> _fused_rms_norm
# Instances: 1. Ops: 41, compute: 4, outputs: 9.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        wait_tensor_871: "bf16[525336576][1]cuda:0",
        arg583_1: "i64[1, 8192][8192, 1]cuda:0",
        wait_tensor_873: "bf16[218112000][1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_default: "bf16[8, 65667072][65667072, 1]cuda:0" = torch.ops.aten.view.default(wait_tensor_871, [8, -1]);  wait_tensor_871 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(view_default, [65667072], 1);  view_default = None
        getitem: "bf16[8, 65667072][65667072, 1]cuda:0" = split_with_sizes_default[0];  split_with_sizes_default = None
        view_dtype: "bf16[8, 65667072][65667072, 1]cuda:0" = torch.ops.aten.view.dtype(getitem, torch.bfloat16);  getitem = None
        view_default_1: "bf16[128256, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(view_dtype, [128256, 4096]);  view_dtype = None

        # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} recompute: MUST_SAVE File: /data/users/bahuang/pytorch/torch/nn/modules/sparse.py:189 in forward, code: return F.embedding(
        embedding_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.embedding.default(view_default_1, arg583_1);  view_default_1 = arg583_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_default_2: "bf16[8, 27264000][27264000, 1]cuda:0" = torch.ops.aten.view.default(wait_tensor_873, [8, -1]);  wait_tensor_873 = None
        split_with_sizes_default_1 = torch.ops.aten.split_with_sizes.default(view_default_2, [512, 2097152, 524288, 524288, 2097152, 512, 7340032, 7340032, 7340032], 1);  view_default_2 = None
        getitem_1: "bf16[8, 512][27264000, 1]cuda:0" = split_with_sizes_default_1[0]
        view_dtype_1: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_1, torch.bfloat16);  getitem_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_1, memory_format = torch.contiguous_format);  view_dtype_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default, [4096]);  clone_default = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'decomposable', 'is_fusible': True, 'partition_id': 3} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        _fused_rms_norm_default = torch.ops.aten._fused_rms_norm.default(embedding_default, [4096], _unsafe_view_default, 1e-05);  embedding_default = _unsafe_view_default = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/normalization.py:427 in forward, code: return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        getitem_2: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = _fused_rms_norm_default[0];  _fused_rms_norm_default = None

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wq', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_3: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_2, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wk', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_2, [8192, 4096])

        # Annotation: {'module_fqn': 'layers.0.attention.qkv_linear.wv', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/pytorch/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        view_default_5: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(getitem_2, [8192, 4096]);  getitem_2 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        getitem_3: "bf16[8, 2097152][27264000, 1]cuda:0" = split_with_sizes_default_1[1]
        view_dtype_2: "bf16[8, 2097152][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_3, torch.bfloat16);  getitem_3 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_1: "bf16[8, 2097152][2097152, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_2, memory_format = torch.contiguous_format);  view_dtype_2 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_1: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_1, [4096, 4096]);  clone_default_1 = None
        getitem_4: "bf16[8, 524288][27264000, 1]cuda:0" = split_with_sizes_default_1[2]
        view_dtype_3: "bf16[8, 524288][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_4, torch.bfloat16);  getitem_4 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_2: "bf16[8, 524288][524288, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_3, memory_format = torch.contiguous_format);  view_dtype_3 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_2: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_2, [1024, 4096]);  clone_default_2 = None
        getitem_5: "bf16[8, 524288][27264000, 1]cuda:0" = split_with_sizes_default_1[3]
        view_dtype_4: "bf16[8, 524288][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_5, torch.bfloat16);  getitem_5 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_3: "bf16[8, 524288][524288, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_4, memory_format = torch.contiguous_format);  view_dtype_4 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_3: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_3, [1024, 4096]);  clone_default_3 = None
        getitem_6: "bf16[8, 2097152][27264000, 1]cuda:0" = split_with_sizes_default_1[4]
        view_dtype_5: "bf16[8, 2097152][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_6, torch.bfloat16);  getitem_6 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_4: "bf16[8, 2097152][2097152, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_5, memory_format = torch.contiguous_format);  view_dtype_5 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_4: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_4, [4096, 4096]);  clone_default_4 = None
        getitem_7: "bf16[8, 512][27264000, 1]cuda:0" = split_with_sizes_default_1[5]
        view_dtype_6: "bf16[8, 512][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_7, torch.bfloat16);  getitem_7 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_5: "bf16[8, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_6, memory_format = torch.contiguous_format);  view_dtype_6 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_5: "bf16[4096][1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_5, [4096]);  clone_default_5 = None
        getitem_8: "bf16[8, 7340032][27264000, 1]cuda:0" = split_with_sizes_default_1[6];  split_with_sizes_default_1 = None
        view_dtype_7: "bf16[8, 7340032][27264000, 1]cuda:0" = torch.ops.aten.view.dtype(getitem_8, torch.bfloat16);  getitem_8 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        clone_default_6: "bf16[8, 7340032][7340032, 1]cuda:0" = torch.ops.aten.clone.default(view_dtype_7, memory_format = torch.contiguous_format);  view_dtype_7 = None

        # Annotation: {'module_fqn': 'layers.0.attention_norm', 'fusion_class': 'view', 'is_fusible': True, 'partition_id': 3} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        _unsafe_view_default_6: "bf16[14336, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_view.default(clone_default_6, [14336, 4096]);  clone_default_6 = None
        return (view_default_3, view_default_4, view_default_5, _unsafe_view_default_1, _unsafe_view_default_2, _unsafe_view_default_3, _unsafe_view_default_4, _unsafe_view_default_5, _unsafe_view_default_6)


def get_inputs():
    wait_tensor_871 = torch.randn([525336576], dtype=torch.bfloat16, device='cuda')
    arg583_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    wait_tensor_873 = torch.randn([218112000], dtype=torch.bfloat16, device='cuda')
    return [wait_tensor_871, arg583_1, wait_tensor_873]

def get_init_inputs():
    return []
