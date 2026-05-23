# Fused region (tok_embeddings): split_with_sizes -> embedding
# Instances: 1. Ops: 6, compute: 2, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        wait_tensor_871: "bf16[525336576][1]cuda:0",
        arg583_1: "i64[1, 8192][8192, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'view', 'is_fusible': True} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1887 in forward, code: output = redistribute_local_tensor(
        view_default: "bf16[8, 65667072][65667072, 1]cuda:0" = torch.ops.aten.view.default(wait_tensor_871, [8, -1]);  wait_tensor_871 = None
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(view_default, [65667072], 1);  view_default = None
        getitem: "bf16[8, 65667072][65667072, 1]cuda:0" = split_with_sizes_default[0];  split_with_sizes_default = None
        view_dtype: "bf16[8, 65667072][65667072, 1]cuda:0" = torch.ops.aten.view.dtype(getitem, torch.bfloat16);  getitem = None
        view_default_1: "bf16[128256, 4096][4096, 1]cuda:0" = torch.ops.aten.view.default(view_dtype, [128256, 4096]);  view_dtype = None

        # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: MUST_SAVE File: /data/users/bahuang/pytorch/torch/nn/modules/sparse.py:189 in forward, code: return F.embedding(
        embedding_default: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0" = torch.ops.aten.embedding.default(view_default_1, arg583_1);  view_default_1 = arg583_1 = None
        return embedding_default


def get_inputs():
    wait_tensor_871 = torch.randn([525336576], dtype=torch.bfloat16, device='cuda')
    arg583_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    return [wait_tensor_871, arg583_1]

def get_init_inputs():
    return []
