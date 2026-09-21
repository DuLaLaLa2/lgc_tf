"""Sliding-window ESM2-3B + LGC + BiMamba 模型。

结构：
    已提取的 Sliding-window ESM2-3B 特征
        -> Linear Projection -> LGC -> BiMamba × N -> MLP
        -> residue-level RNA-binding logits

ESM2 特征已经在预处理阶段通过 sliding-window 提取，本模型不会运行 ESM2。
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 仅用于直接运行嵌套目录中的脚本；不修改原项目代码。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lgc import LocalGeometricContext
from src.models.sde import SpatialDistanceEncoder

from bimamba_block import BiMambaBlock


class ESM2LGCBiMamba(nn.Module):
    """完整残基级二分类模型。

    输入：
        x[B,L,2560]、coords[B,L,3]、mask/coord_mask[B,L]、lengths[B]。
    输出：
        logits[B,L]；padding 位置为 0，训练与指标计算仍须使用 mask 排除。
    """

    def __init__(
        self,
        esm2_dim: int = 2560,
        d_model: int = 256,
        num_bimamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        sde_dim: int = 16,
        cutoff: float = 30.0,
        k_neighbors: int = 16,
        mamba_backend: str = "mamba",
    ):
        super().__init__()
        self.esm2_dim = esm2_dim
        self.d_model = d_model

        # 保留基线投影中的 GELU，避免除全局建模模块外再改变控制变量。
        self.input_projection = nn.Sequential(
            nn.Linear(esm2_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_encoder = SpatialDistanceEncoder(num_basis=sde_dim, cutoff=cutoff)
        self.lgc = LocalGeometricContext(
            hidden_dim=d_model,
            sde_dim=sde_dim,
            k_neighbors=k_neighbors,
            dropout=dropout,
        )
        self.post_lgc_norm = nn.LayerNorm(d_model)
        self.bimamba_layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    backend=mamba_backend,
                )
                for _ in range(num_bimamba_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def run_bimamba_by_length(
        self,
        h: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """逐蛋白运行 BiMamba，避免反转操作混入 padding。

        输入：h[B,Lmax,D] 和真实长度 lengths[B]。
        输出：补齐后的全局特征 [B,Lmax,D]，padding 部分为 0。
        """

        batch_size, padded_length, _ = h.shape
        length_values = lengths.detach().cpu().tolist()
        if len(length_values) != batch_size:
            raise ValueError("lengths 数量与 batch size 不一致")

        full_length_outputs = []
        for batch_index, length_value in enumerate(length_values):
            length = int(length_value)
            if length < 1 or length > padded_length:
                raise ValueError(f"非法真实长度 {length}，padding 长度为 {padded_length}")
            sequence_features = self.post_lgc_norm(h[batch_index : batch_index + 1, :length])
            for layer in self.bimamba_layers:
                sequence_features = layer(sequence_features)
            sequence_features = self.final_norm(sequence_features)
            full_length_outputs.append(
                F.pad(sequence_features, (0, 0, 0, padded_length - length))
            )
        return torch.cat(full_length_outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        coord_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.esm2_dim:
            raise ValueError(f"x 应为 [B,L,{self.esm2_dim}]，实际 {tuple(x.shape)}")
        if lengths is None:
            lengths = mask.sum(dim=1)

        h = self.input_projection(x)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        spatial = self.spatial_encoder(coords=coords, mask=mask, coord_mask=coord_mask)
        h = self.lgc(
            h=h,
            dist=spatial["dist"],
            sde=spatial["sde"],
            mask=mask,
            sde_mask=spatial["sde_mask"],
            return_knn=False,
        )["h"]

        h = self.run_bimamba_by_length(h, lengths)
        logits = self.classifier(h).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), 0.0)
        assert logits.shape == mask.shape
        assert logits.shape[-1] == x.shape[1]
        return logits

