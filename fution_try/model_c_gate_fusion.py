import torch
import torch.nn as nn

from src.models.lgc import LocalGeometricContext
from src.models.lgc_transformer_model import SinusoidalSequencePositionalEncoding
from src.models.sde import SpatialDistanceEncoder


class LGCTransformerGateFusion(nn.Module):
    """Experiment C: h_final = gate * h_tf + (1 - gate) * h_lgc."""

    method_name = "C_gate_fusion"

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
        use_seq_pe: bool = True,
    ):
        super().__init__()
        self.use_seq_pe = use_seq_pe
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_encoder = SpatialDistanceEncoder(num_basis=sde_dim, cutoff=cutoff)
        self.lgc = LocalGeometricContext(hidden_dim=hidden_dim, sde_dim=sde_dim, k_neighbors=k_neighbors, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.seq_pe = SinusoidalSequencePositionalEncoding(hidden_dim)
        self.pos_dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def add_seq_pe(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_seq_pe:
            seq_pe = self.seq_pe(length=h.shape[1], device=h.device, dtype=h.dtype)
            h = h + seq_pe.expand(h.shape[0], -1, -1)
        h = self.pos_dropout(h)
        return h * mask.unsqueeze(-1).to(h.dtype)

    def forward(self, x, coords, mask, coord_mask):
        h = self.input_proj(x)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        spatial_outputs = self.spatial_encoder(coords=coords, mask=mask, coord_mask=coord_mask)
        lgc_outputs = self.lgc(
            h=h,
            dist=spatial_outputs["dist"],
            sde=spatial_outputs["sde"],
            mask=mask,
            sde_mask=spatial_outputs["sde_mask"],
            return_knn=False,
        )
        h_lgc = lgc_outputs["h"]
        h_tf = self.encoder(self.add_seq_pe(h_lgc, mask), src_key_padding_mask=~mask.bool())
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h_lgc, h_tf], dim=-1)))
        h_final = gate * h_tf + (1.0 - gate) * h_lgc
        h_final = h_final * mask.unsqueeze(-1).to(h_final.dtype)
        return self.classifier(h_final).squeeze(-1)

