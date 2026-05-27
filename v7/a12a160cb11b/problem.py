# Fused region (tok_embeddings): embedding_dense_backward -> _to_copy
# Instances: 1. Ops: 2, compute: 2, outputs: 1.

import torch
import torch.nn as nn
from torch import device  # print_readable emits bare device(...) calls
from math import inf, nan

class Model(torch.nn.Module):
    def forward(
        self,
        add_223: "bf16[1, 8192, 4096][33554432, 4096, 1]cuda:0",
        arg583_1: "i64[1, 8192][8192, 1]cuda:0",
    ):
        # autograd_backward: True # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 2} File: /data/users/bahuang/pytorch/torch/nn/modules/sparse.py:189 in forward, code: return F.embedding(
        embedding_dense_backward_default: "bf16[128256, 4096][4096, 1]cuda:0" = torch.ops.aten.embedding_dense_backward.default(add_223, arg583_1, 128256, -1, False);  add_223 = arg583_1 = None

        # autograd_backward: True # Annotation: {'module_fqn': 'tok_embeddings', 'fusion_class': 'pointwise', 'is_fusible': True, 'partition_id': 2} File: /data/users/bahuang/pytorch/torch/distributed/tensor/_redistribute.py:1865 in forward, code: local_tensor = input._local_tensor.to(dtype=op_dtype)
        _to_copy_default: "f32[128256, 4096][4096, 1]cuda:0" = torch.ops.aten._to_copy.default(embedding_dense_backward_default, dtype = torch.float32);  embedding_dense_backward_default = None
        return _to_copy_default


def get_inputs():
    add_223 = torch.randn([1, 8192, 4096], dtype=torch.bfloat16, device='cuda')
    arg583_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int64, device='cuda')
    return [add_223, arg583_1]

def get_init_inputs():
    return []
