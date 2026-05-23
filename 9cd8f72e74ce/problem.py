# Fused region (layers.*.attention): mul
# Instances: 32. Ops: 2, compute: 1, outputs: 1.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        _conj_63: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0",
        view_as_complex_127: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        clone_default: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.clone.default(_conj_63);  _conj_63 = None
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_127, clone_default);  view_as_complex_127 = clone_default = None
        return mul_tensor


def get_inputs():
    _conj_63 = torch.randn([1, 8192, 1, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_127 = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    return [_conj_63, view_as_complex_127]

def get_init_inputs():
    return []
