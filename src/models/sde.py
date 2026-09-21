import math
from typing import Optional, Dict

import torch
import torch.nn as nn


def build_pair_mask(
    mask: torch.Tensor,
    coord_mask: Optional[torch.Tensor] = None,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Build valid residue-pair mask.

    Args:
        mask:
            [B, L], True for real residues, False for padding.
        coord_mask:
            [B, L], True for residues with valid coordinates.
        include_self:
            Whether to include i == j pairs.

    Returns:
        pair_mask:
            [B, L, L]
    """
    valid = mask.bool()

    if coord_mask is not None:
        valid = valid & coord_mask.bool()

    pair_mask = valid[:, :, None] & valid[:, None, :]

    if not include_self:
        B, L = mask.shape
        eye = torch.eye(L, dtype=torch.bool, device=mask.device).unsqueeze(0)
        pair_mask = pair_mask & (~eye)

    return pair_mask


def pairwise_ca_distance(
    coords: torch.Tensor,
    mask: torch.Tensor,
    coord_mask: Optional[torch.Tensor] = None,
    include_self: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute pairwise C-alpha distance matrix.

    Args:
        coords:
            [B, L, 3]
        mask:
            [B, L]
        coord_mask:
            [B, L]
        include_self:
            Whether pair_mask includes diagonal pairs.

    Returns:
        {
            "dist": [B, L, L],
            "pair_mask": [B, L, L]
        }
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [B, L, 3], got {coords.shape}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be [B, L], got {mask.shape}")

    B, L, _ = coords.shape

    pair_mask = build_pair_mask(
        mask=mask,
        coord_mask=coord_mask,
        include_self=include_self,
    )

    coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

    diff = coords[:, :, None, :] - coords[:, None, :, :]   # [B, L, L, 3]
    dist = torch.linalg.norm(diff, dim=-1)                 # [B, L, L]

    dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    dist = dist.masked_fill(~pair_mask, 0.0)

    return {
        "dist": dist,
        "pair_mask": pair_mask,
    }


class BesselSDE(nn.Module):
    """
    Spatial Distance Encoding using Bessel basis functions.

    For each distance r, compute:

        b_n(r) = sqrt(2 / c) * sin((n + 1) * pi * r / c) / (r + eps)

    Args:
        num_basis:
            Number of Bessel basis functions K.
        cutoff:
            Distance cutoff c, usually 30 Angstrom.
        eps:
            Small value to avoid division by zero.
        zero_self:
            Whether to set diagonal i == j encoding to zero.
    """

    def __init__(
        self,
        num_basis: int = 16,
        cutoff: float = 30.0,
        eps: float = 1e-6,
        zero_self: bool = True,
    ):
        super().__init__()

        self.num_basis = num_basis
        self.cutoff = float(cutoff)
        self.eps = eps
        self.zero_self = zero_self

        freq = torch.arange(1, num_basis + 1, dtype=torch.float32)
        freq = freq * math.pi / self.cutoff

        self.register_buffer("freq", freq, persistent=False)
        self.scale = math.sqrt(2.0 / self.cutoff)

    def forward(
        self,
        dist: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            dist:
                [B, L, L]
            pair_mask:
                [B, L, L], True for valid residue pairs.

        Returns:
            {
                "sde": [B, L, L, K],
                "sde_mask": [B, L, L]
            }
        """
        if dist.ndim != 3:
            raise ValueError(f"dist must be [B, L, L], got {dist.shape}")

        B, L, L2 = dist.shape
        if L != L2:
            raise ValueError(f"dist must be square, got {dist.shape}")

        r = torch.clamp(dist, min=0.0, max=self.cutoff)

        # [B, L, L, 1] * [K] -> [B, L, L, K]
        r_expand = r.unsqueeze(-1)
        sde = self.scale * torch.sin(r_expand * self.freq) / (r_expand + self.eps)

        # Valid SDE region:
        #   1. valid pair
        #   2. distance > eps
        #   3. distance < cutoff
        sde_mask = (dist > self.eps) & (dist < self.cutoff)

        if pair_mask is not None:
            sde_mask = sde_mask & pair_mask.bool()

        if self.zero_self:
            eye = torch.eye(L, dtype=torch.bool, device=dist.device).unsqueeze(0)
            sde_mask = sde_mask & (~eye)

        sde = sde * sde_mask.unsqueeze(-1).to(sde.dtype)
        sde = torch.nan_to_num(sde, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "sde": sde,
            "sde_mask": sde_mask,
        }


class SpatialDistanceEncoder(nn.Module):
    """
    Complete Stage 3 encoder:

        coords + mask -> dist + pair_mask + sde + sde_mask
    """

    def __init__(
        self,
        num_basis: int = 16,
        cutoff: float = 30.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.sde = BesselSDE(
            num_basis=num_basis,
            cutoff=cutoff,
            eps=eps,
            zero_self=True,
        )

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        coord_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            coords:
                [B, L, 3]
            mask:
                [B, L]
            coord_mask:
                [B, L]

        Returns:
            {
                "dist":      [B, L, L],
                "pair_mask": [B, L, L],
                "sde":       [B, L, L, K],
                "sde_mask":  [B, L, L]
            }
        """
        distance_outputs = pairwise_ca_distance(
            coords=coords,
            mask=mask,
            coord_mask=coord_mask,
            include_self=True,
        )

        dist = distance_outputs["dist"]
        pair_mask = distance_outputs["pair_mask"]

        sde_outputs = self.sde(
            dist=dist,
            pair_mask=pair_mask,
        )

        return {
            "dist": dist,
            "pair_mask": pair_mask,
            "sde": sde_outputs["sde"],
            "sde_mask": sde_outputs["sde_mask"],
        }