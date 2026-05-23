# Fused region (layers.*.attention): index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul -> index -> mul -> mul
# Instances: 1. Ops: 165, compute: 99, outputs: 66.

import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(
        self,
        arg586_1: "i32[1, 8192][8192, 1]cuda:0",
        arg582_1: "c64[8192, 64][64, 1]cuda:0",
        view_as_complex_62: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_63: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_62_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_63_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_60_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_61_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_58_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_59_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_56_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_57_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_54_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_55_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_52_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_53_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_50_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_51_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_48_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_49_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_46_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_47_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_44_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_45_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_42_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_43_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_40_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_41_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_38_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_39_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_36_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_37_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_34_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_35_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_32_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_33_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_30_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_31_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_28_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_29_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_26_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_27_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_24_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_25_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_22_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_23_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_20_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_21_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_18_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_19_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_16_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_17_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_14_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_15_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_12_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_13_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_10_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_11_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_8_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_9_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_6_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_7_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_4_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_5_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_2_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_3_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
        view_as_complex_recomputed: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0",
        view_as_complex_1_recomputed: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0",
    ):
        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim]);  squeeze_dim = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor, [1, 8192, 1, 64]);  index_tensor = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_62, reshape_default);  view_as_complex_62 = None
        mul_tensor_1: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_63, reshape_default);  view_as_complex_63 = reshape_default = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_1: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_1: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_1]);  squeeze_dim_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_1: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_1, [1, 8192, 1, 64]);  index_tensor_1 = None

        # Annotation: {'module_fqn': 'layers.31.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_2: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_62_recomputed, reshape_default_1);  view_as_complex_62_recomputed = None
        mul_tensor_3: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_63_recomputed, reshape_default_1);  view_as_complex_63_recomputed = reshape_default_1 = None

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_2: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_2: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_2]);  squeeze_dim_2 = None

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_2: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_2, [1, 8192, 1, 64]);  index_tensor_2 = None

        # Annotation: {'module_fqn': 'layers.30.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_4: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_60_recomputed, reshape_default_2);  view_as_complex_60_recomputed = None
        mul_tensor_5: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_61_recomputed, reshape_default_2);  view_as_complex_61_recomputed = reshape_default_2 = None

        # Annotation: {'module_fqn': 'layers.29.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_3: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.29.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_3: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_3]);  squeeze_dim_3 = None

        # Annotation: {'module_fqn': 'layers.29.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_3: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_3, [1, 8192, 1, 64]);  index_tensor_3 = None

        # Annotation: {'module_fqn': 'layers.29.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_6: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_58_recomputed, reshape_default_3);  view_as_complex_58_recomputed = None
        mul_tensor_7: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_59_recomputed, reshape_default_3);  view_as_complex_59_recomputed = reshape_default_3 = None

        # Annotation: {'module_fqn': 'layers.28.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_4: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.28.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_4: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_4]);  squeeze_dim_4 = None

        # Annotation: {'module_fqn': 'layers.28.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_4: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_4, [1, 8192, 1, 64]);  index_tensor_4 = None

        # Annotation: {'module_fqn': 'layers.28.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_8: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_56_recomputed, reshape_default_4);  view_as_complex_56_recomputed = None
        mul_tensor_9: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_57_recomputed, reshape_default_4);  view_as_complex_57_recomputed = reshape_default_4 = None

        # Annotation: {'module_fqn': 'layers.27.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_5: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.27.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_5: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_5]);  squeeze_dim_5 = None

        # Annotation: {'module_fqn': 'layers.27.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_5: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_5, [1, 8192, 1, 64]);  index_tensor_5 = None

        # Annotation: {'module_fqn': 'layers.27.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_10: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_54_recomputed, reshape_default_5);  view_as_complex_54_recomputed = None
        mul_tensor_11: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_55_recomputed, reshape_default_5);  view_as_complex_55_recomputed = reshape_default_5 = None

        # Annotation: {'module_fqn': 'layers.26.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_6: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.26.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_6: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_6]);  squeeze_dim_6 = None

        # Annotation: {'module_fqn': 'layers.26.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_6: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_6, [1, 8192, 1, 64]);  index_tensor_6 = None

        # Annotation: {'module_fqn': 'layers.26.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_12: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_52_recomputed, reshape_default_6);  view_as_complex_52_recomputed = None
        mul_tensor_13: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_53_recomputed, reshape_default_6);  view_as_complex_53_recomputed = reshape_default_6 = None

        # Annotation: {'module_fqn': 'layers.25.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_7: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.25.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_7: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_7]);  squeeze_dim_7 = None

        # Annotation: {'module_fqn': 'layers.25.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_7: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_7, [1, 8192, 1, 64]);  index_tensor_7 = None

        # Annotation: {'module_fqn': 'layers.25.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_14: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_50_recomputed, reshape_default_7);  view_as_complex_50_recomputed = None
        mul_tensor_15: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_51_recomputed, reshape_default_7);  view_as_complex_51_recomputed = reshape_default_7 = None

        # Annotation: {'module_fqn': 'layers.24.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_8: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.24.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_8: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_8]);  squeeze_dim_8 = None

        # Annotation: {'module_fqn': 'layers.24.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_8: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_8, [1, 8192, 1, 64]);  index_tensor_8 = None

        # Annotation: {'module_fqn': 'layers.24.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_16: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_48_recomputed, reshape_default_8);  view_as_complex_48_recomputed = None
        mul_tensor_17: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_49_recomputed, reshape_default_8);  view_as_complex_49_recomputed = reshape_default_8 = None

        # Annotation: {'module_fqn': 'layers.23.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_9: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.23.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_9: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_9]);  squeeze_dim_9 = None

        # Annotation: {'module_fqn': 'layers.23.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_9: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_9, [1, 8192, 1, 64]);  index_tensor_9 = None

        # Annotation: {'module_fqn': 'layers.23.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_18: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_46_recomputed, reshape_default_9);  view_as_complex_46_recomputed = None
        mul_tensor_19: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_47_recomputed, reshape_default_9);  view_as_complex_47_recomputed = reshape_default_9 = None

        # Annotation: {'module_fqn': 'layers.22.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_10: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.22.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_10: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_10]);  squeeze_dim_10 = None

        # Annotation: {'module_fqn': 'layers.22.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_10: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_10, [1, 8192, 1, 64]);  index_tensor_10 = None

        # Annotation: {'module_fqn': 'layers.22.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_20: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_44_recomputed, reshape_default_10);  view_as_complex_44_recomputed = None
        mul_tensor_21: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_45_recomputed, reshape_default_10);  view_as_complex_45_recomputed = reshape_default_10 = None

        # Annotation: {'module_fqn': 'layers.21.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_11: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.21.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_11: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_11]);  squeeze_dim_11 = None

        # Annotation: {'module_fqn': 'layers.21.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_11: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_11, [1, 8192, 1, 64]);  index_tensor_11 = None

        # Annotation: {'module_fqn': 'layers.21.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_22: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_42_recomputed, reshape_default_11);  view_as_complex_42_recomputed = None
        mul_tensor_23: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_43_recomputed, reshape_default_11);  view_as_complex_43_recomputed = reshape_default_11 = None

        # Annotation: {'module_fqn': 'layers.20.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_12: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.20.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_12: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_12]);  squeeze_dim_12 = None

        # Annotation: {'module_fqn': 'layers.20.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_12: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_12, [1, 8192, 1, 64]);  index_tensor_12 = None

        # Annotation: {'module_fqn': 'layers.20.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_24: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_40_recomputed, reshape_default_12);  view_as_complex_40_recomputed = None
        mul_tensor_25: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_41_recomputed, reshape_default_12);  view_as_complex_41_recomputed = reshape_default_12 = None

        # Annotation: {'module_fqn': 'layers.19.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_13: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.19.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_13: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_13]);  squeeze_dim_13 = None

        # Annotation: {'module_fqn': 'layers.19.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_13: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_13, [1, 8192, 1, 64]);  index_tensor_13 = None

        # Annotation: {'module_fqn': 'layers.19.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_26: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_38_recomputed, reshape_default_13);  view_as_complex_38_recomputed = None
        mul_tensor_27: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_39_recomputed, reshape_default_13);  view_as_complex_39_recomputed = reshape_default_13 = None

        # Annotation: {'module_fqn': 'layers.18.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_14: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.18.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_14: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_14]);  squeeze_dim_14 = None

        # Annotation: {'module_fqn': 'layers.18.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_14: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_14, [1, 8192, 1, 64]);  index_tensor_14 = None

        # Annotation: {'module_fqn': 'layers.18.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_28: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_36_recomputed, reshape_default_14);  view_as_complex_36_recomputed = None
        mul_tensor_29: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_37_recomputed, reshape_default_14);  view_as_complex_37_recomputed = reshape_default_14 = None

        # Annotation: {'module_fqn': 'layers.17.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_15: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.17.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_15: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_15]);  squeeze_dim_15 = None

        # Annotation: {'module_fqn': 'layers.17.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_15: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_15, [1, 8192, 1, 64]);  index_tensor_15 = None

        # Annotation: {'module_fqn': 'layers.17.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_30: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_34_recomputed, reshape_default_15);  view_as_complex_34_recomputed = None
        mul_tensor_31: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_35_recomputed, reshape_default_15);  view_as_complex_35_recomputed = reshape_default_15 = None

        # Annotation: {'module_fqn': 'layers.16.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_16: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.16.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_16: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_16]);  squeeze_dim_16 = None

        # Annotation: {'module_fqn': 'layers.16.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_16: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_16, [1, 8192, 1, 64]);  index_tensor_16 = None

        # Annotation: {'module_fqn': 'layers.16.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_32: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_32_recomputed, reshape_default_16);  view_as_complex_32_recomputed = None
        mul_tensor_33: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_33_recomputed, reshape_default_16);  view_as_complex_33_recomputed = reshape_default_16 = None

        # Annotation: {'module_fqn': 'layers.15.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_17: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.15.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_17: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_17]);  squeeze_dim_17 = None

        # Annotation: {'module_fqn': 'layers.15.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_17: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_17, [1, 8192, 1, 64]);  index_tensor_17 = None

        # Annotation: {'module_fqn': 'layers.15.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_34: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_30_recomputed, reshape_default_17);  view_as_complex_30_recomputed = None
        mul_tensor_35: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_31_recomputed, reshape_default_17);  view_as_complex_31_recomputed = reshape_default_17 = None

        # Annotation: {'module_fqn': 'layers.14.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_18: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.14.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_18: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_18]);  squeeze_dim_18 = None

        # Annotation: {'module_fqn': 'layers.14.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_18: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_18, [1, 8192, 1, 64]);  index_tensor_18 = None

        # Annotation: {'module_fqn': 'layers.14.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_36: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_28_recomputed, reshape_default_18);  view_as_complex_28_recomputed = None
        mul_tensor_37: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_29_recomputed, reshape_default_18);  view_as_complex_29_recomputed = reshape_default_18 = None

        # Annotation: {'module_fqn': 'layers.13.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_19: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.13.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_19: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_19]);  squeeze_dim_19 = None

        # Annotation: {'module_fqn': 'layers.13.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_19: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_19, [1, 8192, 1, 64]);  index_tensor_19 = None

        # Annotation: {'module_fqn': 'layers.13.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_38: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_26_recomputed, reshape_default_19);  view_as_complex_26_recomputed = None
        mul_tensor_39: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_27_recomputed, reshape_default_19);  view_as_complex_27_recomputed = reshape_default_19 = None

        # Annotation: {'module_fqn': 'layers.12.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_20: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.12.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_20: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_20]);  squeeze_dim_20 = None

        # Annotation: {'module_fqn': 'layers.12.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_20: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_20, [1, 8192, 1, 64]);  index_tensor_20 = None

        # Annotation: {'module_fqn': 'layers.12.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_40: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_24_recomputed, reshape_default_20);  view_as_complex_24_recomputed = None
        mul_tensor_41: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_25_recomputed, reshape_default_20);  view_as_complex_25_recomputed = reshape_default_20 = None

        # Annotation: {'module_fqn': 'layers.11.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_21: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.11.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_21: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_21]);  squeeze_dim_21 = None

        # Annotation: {'module_fqn': 'layers.11.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_21: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_21, [1, 8192, 1, 64]);  index_tensor_21 = None

        # Annotation: {'module_fqn': 'layers.11.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_42: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_22_recomputed, reshape_default_21);  view_as_complex_22_recomputed = None
        mul_tensor_43: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_23_recomputed, reshape_default_21);  view_as_complex_23_recomputed = reshape_default_21 = None

        # Annotation: {'module_fqn': 'layers.10.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_22: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.10.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_22: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_22]);  squeeze_dim_22 = None

        # Annotation: {'module_fqn': 'layers.10.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_22: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_22, [1, 8192, 1, 64]);  index_tensor_22 = None

        # Annotation: {'module_fqn': 'layers.10.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_44: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_20_recomputed, reshape_default_22);  view_as_complex_20_recomputed = None
        mul_tensor_45: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_21_recomputed, reshape_default_22);  view_as_complex_21_recomputed = reshape_default_22 = None

        # Annotation: {'module_fqn': 'layers.9.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_23: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.9.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_23: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_23]);  squeeze_dim_23 = None

        # Annotation: {'module_fqn': 'layers.9.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_23: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_23, [1, 8192, 1, 64]);  index_tensor_23 = None

        # Annotation: {'module_fqn': 'layers.9.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_46: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_18_recomputed, reshape_default_23);  view_as_complex_18_recomputed = None
        mul_tensor_47: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_19_recomputed, reshape_default_23);  view_as_complex_19_recomputed = reshape_default_23 = None

        # Annotation: {'module_fqn': 'layers.8.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_24: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.8.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_24: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_24]);  squeeze_dim_24 = None

        # Annotation: {'module_fqn': 'layers.8.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_24: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_24, [1, 8192, 1, 64]);  index_tensor_24 = None

        # Annotation: {'module_fqn': 'layers.8.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_48: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_16_recomputed, reshape_default_24);  view_as_complex_16_recomputed = None
        mul_tensor_49: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_17_recomputed, reshape_default_24);  view_as_complex_17_recomputed = reshape_default_24 = None

        # Annotation: {'module_fqn': 'layers.7.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_25: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.7.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_25: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_25]);  squeeze_dim_25 = None

        # Annotation: {'module_fqn': 'layers.7.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_25: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_25, [1, 8192, 1, 64]);  index_tensor_25 = None

        # Annotation: {'module_fqn': 'layers.7.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_50: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_14_recomputed, reshape_default_25);  view_as_complex_14_recomputed = None
        mul_tensor_51: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_15_recomputed, reshape_default_25);  view_as_complex_15_recomputed = reshape_default_25 = None

        # Annotation: {'module_fqn': 'layers.6.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_26: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.6.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_26: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_26]);  squeeze_dim_26 = None

        # Annotation: {'module_fqn': 'layers.6.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_26: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_26, [1, 8192, 1, 64]);  index_tensor_26 = None

        # Annotation: {'module_fqn': 'layers.6.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_52: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_12_recomputed, reshape_default_26);  view_as_complex_12_recomputed = None
        mul_tensor_53: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_13_recomputed, reshape_default_26);  view_as_complex_13_recomputed = reshape_default_26 = None

        # Annotation: {'module_fqn': 'layers.5.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_27: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.5.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_27: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_27]);  squeeze_dim_27 = None

        # Annotation: {'module_fqn': 'layers.5.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_27: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_27, [1, 8192, 1, 64]);  index_tensor_27 = None

        # Annotation: {'module_fqn': 'layers.5.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_54: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_10_recomputed, reshape_default_27);  view_as_complex_10_recomputed = None
        mul_tensor_55: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_11_recomputed, reshape_default_27);  view_as_complex_11_recomputed = reshape_default_27 = None

        # Annotation: {'module_fqn': 'layers.4.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_28: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.4.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_28: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_28]);  squeeze_dim_28 = None

        # Annotation: {'module_fqn': 'layers.4.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_28: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_28, [1, 8192, 1, 64]);  index_tensor_28 = None

        # Annotation: {'module_fqn': 'layers.4.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_56: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_8_recomputed, reshape_default_28);  view_as_complex_8_recomputed = None
        mul_tensor_57: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_9_recomputed, reshape_default_28);  view_as_complex_9_recomputed = reshape_default_28 = None

        # Annotation: {'module_fqn': 'layers.3.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_29: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.3.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_29: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_29]);  squeeze_dim_29 = None

        # Annotation: {'module_fqn': 'layers.3.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_29: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_29, [1, 8192, 1, 64]);  index_tensor_29 = None

        # Annotation: {'module_fqn': 'layers.3.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_58: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_6_recomputed, reshape_default_29);  view_as_complex_6_recomputed = None
        mul_tensor_59: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_7_recomputed, reshape_default_29);  view_as_complex_7_recomputed = reshape_default_29 = None

        # Annotation: {'module_fqn': 'layers.2.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_30: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.2.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_30: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_30]);  squeeze_dim_30 = None

        # Annotation: {'module_fqn': 'layers.2.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_30: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_30, [1, 8192, 1, 64]);  index_tensor_30 = None

        # Annotation: {'module_fqn': 'layers.2.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_60: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_4_recomputed, reshape_default_30);  view_as_complex_4_recomputed = None
        mul_tensor_61: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_5_recomputed, reshape_default_30);  view_as_complex_5_recomputed = reshape_default_30 = None

        # Annotation: {'module_fqn': 'layers.1.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_31: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0)

        # Annotation: {'module_fqn': 'layers.1.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_31: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_31]);  squeeze_dim_31 = None

        # Annotation: {'module_fqn': 'layers.1.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_31: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_31, [1, 8192, 1, 64]);  index_tensor_31 = None

        # Annotation: {'module_fqn': 'layers.1.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_62: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_2_recomputed, reshape_default_31);  view_as_complex_2_recomputed = None
        mul_tensor_63: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_3_recomputed, reshape_default_31);  view_as_complex_3_recomputed = reshape_default_31 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        squeeze_dim_32: "i32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(arg586_1, 0);  arg586_1 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        index_tensor_32: "c64[8192, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(arg582_1, [squeeze_dim_32]);  arg582_1 = squeeze_dim_32 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'view', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        reshape_default_32: "c64[1, 8192, 1, 64][524288, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(index_tensor_32, [1, 8192, 1, 64]);  index_tensor_32 = None

        # Annotation: {'module_fqn': 'layers.0.attention', 'fusion_class': 'pointwise', 'is_fusible': True} recompute: PREFER_RECOMPUTE File: /data/users/bahuang/torchtitan/torchtitan/models/common/attention.py:657 in forward, code: xq, xk = apply_rotary_emb_complex(
        mul_tensor_64: "c64[1, 8192, 32, 64][16777216, 2048, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_recomputed, reshape_default_32);  view_as_complex_recomputed = None
        mul_tensor_65: "c64[1, 8192, 8, 64][4194304, 512, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_as_complex_1_recomputed, reshape_default_32);  view_as_complex_1_recomputed = reshape_default_32 = None
        return (mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, mul_tensor_26, mul_tensor_27, mul_tensor_28, mul_tensor_29, mul_tensor_30, mul_tensor_31, mul_tensor_32, mul_tensor_33, mul_tensor_34, mul_tensor_35, mul_tensor_36, mul_tensor_37, mul_tensor_38, mul_tensor_39, mul_tensor_40, mul_tensor_41, mul_tensor_42, mul_tensor_43, mul_tensor_44, mul_tensor_45, mul_tensor_46, mul_tensor_47, mul_tensor_48, mul_tensor_49, mul_tensor_50, mul_tensor_51, mul_tensor_52, mul_tensor_53, mul_tensor_54, mul_tensor_55, mul_tensor_56, mul_tensor_57, mul_tensor_58, mul_tensor_59, mul_tensor_60, mul_tensor_61, mul_tensor_62, mul_tensor_63, mul_tensor_64, mul_tensor_65)


def get_inputs():
    arg586_1 = torch.randint(0, 100, [1, 8192], dtype=torch.int32, device='cuda')
    arg582_1 = torch.randn([8192, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_62 = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_63 = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_62_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_63_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_60_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_61_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_58_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_59_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_56_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_57_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_54_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_55_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_52_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_53_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_50_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_51_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_48_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_49_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_46_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_47_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_44_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_45_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_42_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_43_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_40_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_41_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_38_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_39_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_36_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_37_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_34_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_35_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_32_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_33_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_30_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_31_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_28_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_29_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_26_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_27_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_24_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_25_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_22_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_23_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_20_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_21_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_18_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_19_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_16_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_17_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_14_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_15_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_12_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_13_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_10_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_11_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_8_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_9_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_6_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_7_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_4_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_5_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_2_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_3_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_recomputed = torch.randn([1, 8192, 32, 64], dtype=torch.complex64, device='cuda')
    view_as_complex_1_recomputed = torch.randn([1, 8192, 8, 64], dtype=torch.complex64, device='cuda')
    return [arg586_1, arg582_1, view_as_complex_62, view_as_complex_63, view_as_complex_62_recomputed, view_as_complex_63_recomputed, view_as_complex_60_recomputed, view_as_complex_61_recomputed, view_as_complex_58_recomputed, view_as_complex_59_recomputed, view_as_complex_56_recomputed, view_as_complex_57_recomputed, view_as_complex_54_recomputed, view_as_complex_55_recomputed, view_as_complex_52_recomputed, view_as_complex_53_recomputed, view_as_complex_50_recomputed, view_as_complex_51_recomputed, view_as_complex_48_recomputed, view_as_complex_49_recomputed, view_as_complex_46_recomputed, view_as_complex_47_recomputed, view_as_complex_44_recomputed, view_as_complex_45_recomputed, view_as_complex_42_recomputed, view_as_complex_43_recomputed, view_as_complex_40_recomputed, view_as_complex_41_recomputed, view_as_complex_38_recomputed, view_as_complex_39_recomputed, view_as_complex_36_recomputed, view_as_complex_37_recomputed, view_as_complex_34_recomputed, view_as_complex_35_recomputed, view_as_complex_32_recomputed, view_as_complex_33_recomputed, view_as_complex_30_recomputed, view_as_complex_31_recomputed, view_as_complex_28_recomputed, view_as_complex_29_recomputed, view_as_complex_26_recomputed, view_as_complex_27_recomputed, view_as_complex_24_recomputed, view_as_complex_25_recomputed, view_as_complex_22_recomputed, view_as_complex_23_recomputed, view_as_complex_20_recomputed, view_as_complex_21_recomputed, view_as_complex_18_recomputed, view_as_complex_19_recomputed, view_as_complex_16_recomputed, view_as_complex_17_recomputed, view_as_complex_14_recomputed, view_as_complex_15_recomputed, view_as_complex_12_recomputed, view_as_complex_13_recomputed, view_as_complex_10_recomputed, view_as_complex_11_recomputed, view_as_complex_8_recomputed, view_as_complex_9_recomputed, view_as_complex_6_recomputed, view_as_complex_7_recomputed, view_as_complex_4_recomputed, view_as_complex_5_recomputed, view_as_complex_2_recomputed, view_as_complex_3_recomputed, view_as_complex_recomputed, view_as_complex_1_recomputed]

def get_init_inputs():
    return []
