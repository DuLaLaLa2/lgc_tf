from typing import Dict

import torch
import torch.nn as nn


class LocalGeometricContext(nn.Module):
    """
    Local Geometric Context (LGC)

    For each residue i:
        1. Select k nearest spatial neighbors according to C-alpha distance.
        2. Use SDE as edge feature.
        3. Aggregate neighbor messages.
        4. Residually inject local geometry into residue feature.

    Input:
        h:        [B, L, D]
        dist:     [B, L, L]
        sde:      [B, L, L, K]
        mask:     [B, L]
        sde_mask: [B, L, L]

    Output:
        h_out:    [B, L, D]
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        sde_dim: int = 16,
        k_neighbors: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.sde_dim = sde_dim
        self.k_neighbors = k_neighbors

        self.node_norm = nn.LayerNorm(hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(sde_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        dist: torch.Tensor,
        sde: torch.Tensor,
        mask: torch.Tensor,
        sde_mask: torch.Tensor,
        return_knn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h:
                [B, L, D]
            dist:
                [B, L, L]
            sde:
                [B, L, L, K]
            mask:
                [B, L]
            sde_mask:
                [B, L, L]
            return_knn:
                whether to return kNN indices and masks.

        Returns:
            {
                "h": [B, L, D],
                "g": [B, L, D],
                optional "knn_idx": [B, L, k],
                optional "knn_mask": [B, L, k]
            }
        """
        if h.ndim != 3:
            raise ValueError(f"h must be [B, L, D], got {h.shape}")

        if dist.ndim != 3:
            raise ValueError(f"dist must be [B, L, L], got {dist.shape}")

        if sde.ndim != 4:
            raise ValueError(f"sde must be [B, L, L, K], got {sde.shape}")

        B, L, D = h.shape
        K = sde.shape[-1]

        if D != self.hidden_dim:
            raise ValueError(f"h hidden dim {D} != expected {self.hidden_dim}")

        if K != self.sde_dim:
            raise ValueError(f"sde dim {K} != expected {self.sde_dim}")

        k_eff = min(self.k_neighbors, L)

        h_norm = self.node_norm(h)

        # Candidate neighbors:
        # sde_mask has already excluded padding, invalid coords, self-pairs, and distances beyond cutoff.
        candidate_mask = sde_mask.bool()

        # Invalid pairs are assigned a very large distance so they will not be selected.
        large_value = torch.finfo(dist.dtype).max / 10.0
        knn_dist = dist.masked_fill(~candidate_mask, large_value)

        # Select k nearest neighbors for each residue.
        # knn_idx: [B, L, k_eff]
        _, knn_idx = torch.topk(
            knn_dist,
            k=k_eff,
            dim=-1,
            largest=False,
        )

        device = h.device

        batch_idx = torch.arange(B, device=device)[:, None, None].expand(B, L, k_eff)
        residue_idx = torch.arange(L, device=device)[None, :, None].expand(B, L, k_eff)

        # Gather neighbor residue features h_j.
        # h_neighbors: [B, L, k_eff, D]
        h_neighbors = h_norm[batch_idx, knn_idx]

        # Gather edge features sde_ij.
        # edge_sde: [B, L, k_eff, K]
        edge_sde = sde[batch_idx, residue_idx, knn_idx]

        # Gather valid neighbor mask.
        # knn_mask: [B, L, k_eff]
        knn_mask = candidate_mask[batch_idx, residue_idx, knn_idx]

        # Edge embedding.
        # edge_feat: [B, L, k_eff, D]
        edge_feat = self.edge_mlp(edge_sde)

        # Message from neighbor j to center residue i.
        # msg_input: [B, L, k_eff, 2D]
        msg_input = torch.cat([h_neighbors, edge_feat], dim=-1)

        # msg: [B, L, k_eff, D]
        msg = self.message_mlp(msg_input)

        msg = msg * knn_mask.unsqueeze(-1).to(msg.dtype)

        # Mean aggregation.
        denom = knn_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(msg.dtype)

        # g: [B, L, D]
        g = msg.sum(dim=2) / denom

        g = self.out_proj(g)

        h_out = h + self.dropout(g)

        # Zero-out padding residues.
        h_out = h_out * mask.unsqueeze(-1).to(h_out.dtype)
        g = g * mask.unsqueeze(-1).to(g.dtype)

        outputs = {
            "h": h_out,
            "g": g,
        }

        if return_knn:
            outputs["knn_idx"] = knn_idx
            outputs["knn_mask"] = knn_mask

        return outputs