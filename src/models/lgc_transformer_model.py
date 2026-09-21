import torch
import torch.nn as nn

from src.models.sde import SpatialDistanceEncoder
from src.models.lgc import LocalGeometricContext
'''
这个模型还不是完整 3D-SRPE，因为还没有 KV 调制注意力，但它可以作为一个重要消融：
    ESM2 + Transformer
    vs
    ESM2 + LGC + Transformer
'''


class SinusoidalSequencePositionalEncoding(nn.Module):
    """
    Traditional sinusoidal sequence positional encoding.

    For residue index pos and channel i:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / self.d_model)
        )

        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)


class ESM2LGCTransformer(nn.Module):
    """
    Stage 4 model:
        ESM2 embedding
        -> Linear projection
        -> SpatialDistanceEncoder
        -> LGC
        -> normal TransformerEncoder
        -> residue classifier

    This model uses 3D geometry through LGC,
    but does not use spatial KV attention yet.
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
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

    def fuse_positional_encodings(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        positional_terms: dict,
    ) -> torch.Tensor:
        """
        Unified positional-encoding fusion.

        Current branch uses only:
            seq_pe: traditional sinusoidal sequence PE

        Future structural or learned PE terms can be added to positional_terms
        and will be summed here before the Transformer.
        """
        for name, pe in positional_terms.items():
            if pe is None:
                continue
            if pe.ndim == 2:
                pe = pe.unsqueeze(0)
            if pe.shape[-2:] != h.shape[-2:]:
                raise ValueError(f"{name} shape {tuple(pe.shape)} is incompatible with h {tuple(h.shape)}")
            h = h + pe.to(device=h.device, dtype=h.dtype)

        h = self.pos_dropout(h)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        return h

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        coord_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [B, L, 2560]
            coords:
                [B, L, 3]
            mask:
                [B, L]
            coord_mask:
                [B, L]

        Returns:
            logits:
                [B, L]
        """
        h = self.input_proj(x)
        h = h * mask.unsqueeze(-1).to(h.dtype)

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

        positional_terms = {}
        if self.use_seq_pe:
            positional_terms["seq_pe"] = self.seq_pe(
                length=h.shape[1],
                device=h.device,
                dtype=h.dtype,
            ).expand(h.shape[0], -1, -1)

        h = self.fuse_positional_encodings(
            h=h,
            mask=mask,
            positional_terms=positional_terms,
        )

        src_key_padding_mask = ~mask.bool()

        h = self.encoder(
            h,
            src_key_padding_mask=src_key_padding_mask,
        )

        logits = self.classifier(h).squeeze(-1)

        return logits
