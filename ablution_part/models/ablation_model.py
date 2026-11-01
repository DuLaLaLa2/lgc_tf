"""消融实验用的可配置模型。

与 src/models/lgc_transformer_model.py 的区别：
    1. 可独立开/关 LGC（3D 局部几何上下文）
    2. 可独立开/关 Sequence Positional Encoding
    3. 可独立开/关 TransformerEncoder（关闭后退化为逐残基分类）
    4. 可配置 k_neighbors，用于研究局部几何感受野对长序列的影响

关闭 LGC 时，模型退化为 ESM2 + (可选 PE) + Transformer，
即不利用任何 3D 结构信号。这正是我们想对比的消融主干。
"""

from typing import Optional

import torch
import torch.nn as nn

# 复用主线实验里的 SDE / LGC 实现（只读，不修改）
from src.models.sde import SpatialDistanceEncoder
from src.models.lgc import LocalGeometricContext


class SinusoidalSequencePositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)


class AblationESM2Model(nn.Module):
    """消融实验模型。

    开关：
        use_lgc:        是否接入 LocalGeometricContext（即 3D 结构信号）
        use_seq_pe:     是否对 Transformer 输入叠加 sinusoidal 位置编码
        use_transformer: 是否使用 TransformerEncoder（关闭后为逐残基分类）
        k_neighbors:    LGC 中的近邻数（仅在 use_lgc=True 时生效）
    """

    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        sde_dim: int = 16,
        cutoff: float = 30.0,
        k_neighbors: int = 16,
        use_lgc: bool = True,
        use_seq_pe: bool = True,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.use_lgc = use_lgc
        self.use_seq_pe = use_seq_pe
        self.use_transformer = use_transformer
        self.k_neighbors = k_neighbors

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_lgc:
            self.spatial_encoder = SpatialDistanceEncoder(
                num_basis=sde_dim,
                cutoff=cutoff,
            )
            self.lgc = LocalGeometricContext(
                hidden_dim=hidden_dim,
                sde_dim=sde_dim,
                k_neighbors=k_neighbors,
                dropout=dropout,
            )
        else:
            self.spatial_encoder = None
            self.lgc = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            if use_transformer
            else None
        )

        self.seq_pe = SinusoidalSequencePositionalEncoding(hidden_dim)
        self.pos_dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        coord_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        h = h * mask.unsqueeze(-1).to(h.dtype)

        if self.use_lgc:
            spatial_outputs = self.spatial_encoder(
                coords=coords,
                mask=mask,
                coord_mask=coord_mask,
            )
            lgc_outputs = self.lgc(
                h=h,
                dist=spatial_outputs["dist"],
                sde=spatial_outputs["sde"],
                mask=mask,
                sde_mask=spatial_outputs["sde_mask"],
                return_knn=False,
            )
            h = lgc_outputs["h"]

        if self.use_seq_pe:
            pe = self.seq_pe(
                length=h.shape[1],
                device=h.device,
                dtype=h.dtype,
            ).expand(h.shape[0], -1, -1)
            h = h + pe

        h = self.pos_dropout(h)
        h = h * mask.unsqueeze(-1).to(h.dtype)

        if self.use_transformer:
            src_key_padding_mask = ~mask.bool()
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        logits = self.classifier(h).squeeze(-1)
        return logits
