from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset
'''
目的：把每个 .pt 文件变成一个训练样本，并把不同长度的蛋白质 padding 成 batch。
'''

def safe_torch_load(path: Path) -> Dict[str, Any]:
    """
    Compatible with different PyTorch versions.
    Some new PyTorch versions use weights_only=True by default.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RNABindingDataset(Dataset):
    """
    Dataset for merged RNA-binding protein features.

    Each .pt file is expected to contain:
        esm2_embedding: [L, 2560]
        ca_coords:      [L, 3]
        coord_mask:     [L]
        label:          [L]
    """

    def __init__(
        self,
        feature_dir: str,
        indices: Optional[List[int]] = None,
        max_len: Optional[int] = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.files = sorted(self.feature_dir.glob("*.pt"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.feature_dir}")

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        data = safe_torch_load(path)

        x = data["esm2_embedding"].float()      # [L, 2560]
        coords = data["ca_coords"].float()      # [L, 3]
        y = data["label"].float()               # [L]

        if "coord_mask" in data:
            coord_mask = data["coord_mask"].bool()
        else:
            coord_mask = torch.isfinite(coords).all(dim=-1)

        seq_id = data.get("id", path.stem)
        sequence = data.get("sequence", "")
        length = int(data.get("length", x.shape[0]))

        if x.ndim != 2:
            raise ValueError(f"{path.name}: esm2_embedding must be [L, D], got {x.shape}")

        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f"{path.name}: ca_coords must be [L, 3], got {coords.shape}")

        if y.ndim != 1:
            raise ValueError(f"{path.name}: label must be [L], got {y.shape}")

        L = x.shape[0]

        if coords.shape[0] != L:
            raise ValueError(f"{path.name}: coords length {coords.shape[0]} != embedding length {L}")

        if y.shape[0] != L:
            raise ValueError(f"{path.name}: label length {y.shape[0]} != embedding length {L}")

        if coord_mask.shape[0] != L:
            raise ValueError(f"{path.name}: coord_mask length {coord_mask.shape[0]} != embedding length {L}")

        if length != L:
            raise ValueError(f"{path.name}: metadata length {length} != actual length {L}")

        if self.max_len is not None and L > self.max_len:
            x = x[:self.max_len]
            coords = coords[:self.max_len]
            coord_mask = coord_mask[:self.max_len]
            y = y[:self.max_len]
            sequence = sequence[:self.max_len]
            L = self.max_len

        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "id": seq_id,
            "file": str(path),
            "sequence": sequence,
            "length": L,
            "x": x,                       # [L, 2560]
            "coords": coords,             # [L, 3]
            "coord_mask": coord_mask,     # [L]
            "y": y,                       # [L]
        }


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-length proteins into one batch.

    Output:
        x:          [B, Lmax, 2560]
        coords:     [B, Lmax, 3]
        coord_mask: [B, Lmax]
        y:          [B, Lmax]
        mask:       [B, Lmax]
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    feat_dim = batch[0]["x"].shape[-1]

    x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    coord_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    y = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    ids = []
    files = []
    lengths = []

    for b, item in enumerate(batch):
        L = item["length"]

        x[b, :L] = item["x"]
        coords[b, :L] = item["coords"]
        coord_mask[b, :L] = item["coord_mask"]
        y[b, :L] = item["y"]
        mask[b, :L] = True

        ids.append(item["id"])
        files.append(item["file"])
        lengths.append(L)

    return {
        "ids": ids,
        "files": files,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "x": x,
        "coords": coords,
        "coord_mask": coord_mask,
        "y": y,
        "mask": mask,
    }