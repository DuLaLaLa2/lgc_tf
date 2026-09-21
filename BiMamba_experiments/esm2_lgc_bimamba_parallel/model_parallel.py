"""LGC 与 BiMamba 并行的残基级预测模型。

                     ┌─ LGC ────────> H_lgc [B,L,d] ──────┐
ESM2 -> Projection ──┤                                    ├─ Concat
                     └─ BiMamba × N -> H_bimamba [B,L,d] ─┘
                                      H_final [B,L,2d] -> MLP -> logits

ESM2 特征已经由 sliding-window 预先提取，本模型不运行 ESM2。
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lgc import LocalGeometricContext
from src.models.sde import SpatialDistanceEncoder

from bimamba_block import BiMambaBlock


class ESM2ParallelLGCBiMamba(nn.Module):
    """从同一投影特征并行提取局部几何与全局序列特征。

    输入：x[B,L,2560]、coords[B,L,3]、mask/coord_mask[B,L]、lengths[B]。
    输出：logits[B,L]。H_final 是未经降维的 [H_lgc || H_bimamba]。
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
        self.bimamba_layers = nn.ModuleList(
            [
                BiMambaBlock(d_model, d_state, d_conv, expand, dropout, mamba_backend)
                for _ in range(num_bimamba_layers)
            ]
        )
        self.bimamba_output_norm = nn.LayerNorm(d_model)
        # H_final 保持 2d，不加 2d->d 投影；分类器直接消费拼接特征。
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def run_bimamba_by_length(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """按真实长度逐蛋白运行 BiMamba，输出补齐后的 [B,Lmax,d]。"""

        batch_size, padded_length, _ = h.shape
        length_values = lengths.detach().cpu().tolist()
        if len(length_values) != batch_size:
            raise ValueError("lengths 数量与 batch size 不一致")
        outputs = []
        for batch_index, length_value in enumerate(length_values):
            length = int(length_value)
            if length < 1 or length > padded_length:
                raise ValueError(f"非法真实长度 {length}，padding 长度={padded_length}")
            sequence_features = h[batch_index : batch_index + 1, :length]
            for layer in self.bimamba_layers:
                sequence_features = layer(sequence_features)
            sequence_features = self.bimamba_output_norm(sequence_features)
            outputs.append(F.pad(sequence_features, (0, 0, 0, padded_length - length)))
        return torch.cat(outputs, dim=0)

    def extract_features(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        coord_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """提取并行特征。

        输出字典包含 h_projected/h_lgc/h_bimamba[B,L,d] 和 h_final[B,L,2d]。
        """

        h_projected = self.input_projection(x)
        h_projected = h_projected * mask.unsqueeze(-1).to(h_projected.dtype)
        spatial = self.spatial_encoder(coords=coords, mask=mask, coord_mask=coord_mask)
        h_lgc = self.lgc(
            h=h_projected,
            dist=spatial["dist"],
            sde=spatial["sde"],
            mask=mask,
            sde_mask=spatial["sde_mask"],
            return_knn=False,
        )["h"]
        h_bimamba = self.run_bimamba_by_length(h_projected, lengths)
        h_final = torch.cat((h_lgc, h_bimamba), dim=-1)
        h_final = h_final * mask.unsqueeze(-1).to(h_final.dtype)
        assert h_lgc.shape == h_bimamba.shape
        assert h_final.shape[-1] == 2 * self.d_model
        return {
            "h_projected": h_projected,
            "h_lgc": h_lgc,
            "h_bimamba": h_bimamba,
            "h_final": h_final,
        }

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
        features = self.extract_features(x, coords, mask, coord_mask, lengths)
        logits = self.classifier(features["h_final"]).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), 0.0)
        assert logits.shape == mask.shape
        return logits
