"""全长合并特征读取器；不运行 ESM2，也不截断长蛋白。"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


def safe_torch_load(path: Path) -> Dict[str, Any]:
    """从磁盘读取一个单蛋白字典，张量统一加载到 CPU。"""

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path}: 期望 dict，实际为 {type(data).__name__}")
    return data


class FullLengthRNADataset(Dataset):
    """读取完整蛋白特征。

    输出单样本包含 x[L,2560]、coords[L,3]、coord_mask[L]、y[L]，
    以及 id、file、sequence、length。任何长度不一致都会立即报错。
    """

    def __init__(self, feature_dir: Path | str):
        self.feature_dir = Path(feature_dir)
        self.files = sorted(self.feature_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"目录中没有 .pt 文件：{self.feature_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        data = safe_torch_load(path)
        missing = {"esm2_embedding", "ca_coords", "label"} - data.keys()
        if missing:
            raise KeyError(f"{path.name}: 缺少字段 {sorted(missing)}")

        x = torch.as_tensor(data["esm2_embedding"]).float()
        coords = torch.as_tensor(data["ca_coords"]).float()
        y = torch.as_tensor(data["label"]).float()
        sample_id = str(data.get("id", path.stem))
        raw_sequence = data.get("sequence", "")
        sequence = "" if raw_sequence is None else str(raw_sequence)

        if x.ndim != 2:
            raise ValueError(f"{sample_id}: esm2_embedding 应为 [L,D]，实际 {tuple(x.shape)}")
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f"{sample_id}: ca_coords 应为 [L,3]，实际 {tuple(coords.shape)}")
        if y.ndim != 1:
            raise ValueError(f"{sample_id}: label 应为 [L]，实际 {tuple(y.shape)}")

        lengths = {
            "esm2_embedding": int(x.shape[0]),
            "ca_coords": int(coords.shape[0]),
            "label": int(y.shape[0]),
            "metadata_length": int(data.get("length", x.shape[0])),
        }
        if sequence:
            lengths["sequence"] = len(sequence)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{sample_id}: 长度不一致，{lengths}，文件={path}")

        length = int(x.shape[0])
        coord_mask = (
            torch.as_tensor(data["coord_mask"]).bool()
            if "coord_mask" in data
            else torch.isfinite(coords).all(dim=-1)
        )
        if coord_mask.ndim != 1 or coord_mask.shape[0] != length:
            raise ValueError(f"{sample_id}: coord_mask 应为 [{length}]，实际 {tuple(coord_mask.shape)}")

        return {
            "id": sample_id,
            "file": str(path),
            "sequence": sequence,
            "length": length,
            "x": x,
            "coords": torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0),
            "coord_mask": coord_mask,
            "y": y,
        }


def pad_collate_full_length(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """补齐变长蛋白，输出 x/coords/y/mask/coord_mask 和真实 lengths。"""

    batch_size = len(batch)
    max_length = max(item["length"] for item in batch)
    feature_dim = batch[0]["x"].shape[-1]
    x = torch.zeros(batch_size, max_length, feature_dim)
    coords = torch.zeros(batch_size, max_length, 3)
    coord_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    y = torch.zeros(batch_size, max_length)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        length = item["length"]
        x[batch_index, :length] = item["x"]
        coords[batch_index, :length] = item["coords"]
        coord_mask[batch_index, :length] = item["coord_mask"]
        y[batch_index, :length] = item["y"]
        mask[batch_index, :length] = True
    return {
        "ids": [item["id"] for item in batch],
        "files": [item["file"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "x": x,
        "coords": coords,
        "coord_mask": coord_mask,
        "y": y,
        "mask": mask,
    }


def has_feature_files(path: Path | None) -> bool:
    """判断指定验证目录是否包含特征文件。"""

    return path is not None and path.is_dir() and any(path.glob("*.pt"))

