"""已有合并特征的只读适配器。

本文件不提取 ESM2，也不截断长序列。一个 ``.pt`` 文件对应一个蛋白质。
"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


def safe_torch_load(path: Path) -> Dict[str, Any]:
    """在 CPU 上读取一个 ``.pt`` 字典。

    输入：特征文件路径。
    输出：至少包含 esm2_embedding、ca_coords、label 的字典。
    """

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path}: 期望读取到 dict，实际为 {type(data).__name__}")
    return data


class FullLengthRNADataset(Dataset):
    """读取完整长度的残基级 ESM2、坐标和标签。

    单样本输出：
        x:          FloatTensor[L, 2560]
        coords:     FloatTensor[L, 3]
        coord_mask: BoolTensor[L]
        y:          FloatTensor[L]
        id/file/sequence/length: 样本元数据

    所有长度在加载时显式校验；本类没有 max_len 参数，因而不会静默截断。
    """

    def __init__(self, feature_dir: Path | str):
        self.feature_dir = Path(feature_dir)
        self.files = sorted(self.feature_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"目录中没有 .pt 特征文件：{self.feature_dir}")

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
        }
        if sequence:
            lengths["sequence"] = len(sequence)
        metadata_length = int(data.get("length", x.shape[0]))
        lengths["metadata_length"] = metadata_length
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{sample_id}: 长度不一致，{lengths}，文件={path}")

        length = int(x.shape[0])
        if "coord_mask" in data:
            coord_mask = torch.as_tensor(data["coord_mask"]).bool()
            if coord_mask.ndim != 1 or coord_mask.shape[0] != length:
                raise ValueError(
                    f"{sample_id}: coord_mask 应为 [{length}]，实际 {tuple(coord_mask.shape)}"
                )
        else:
            coord_mask = torch.isfinite(coords).all(dim=-1)

        # 无效坐标由 coord_mask 保留语义；数值本身归零，防止 NaN 传播。
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "id": sample_id,
            "file": str(path),
            "sequence": sequence,
            "length": length,
            "x": x,
            "coords": coords,
            "coord_mask": coord_mask,
            "y": y,
        }


def pad_collate_full_length(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把变长蛋白补齐为 batch，但保留每条蛋白的真实长度。

    输入：FullLengthRNADataset 返回的样本列表。
    输出：x[B,Lmax,D]、coords[B,Lmax,3]、y/mask/coord_mask[B,Lmax]，
    以及 ids、files、sequences、lengths[B]。mask=True 表示真实残基。
    """

    batch_size = len(batch)
    max_length = max(item["length"] for item in batch)
    feature_dim = batch[0]["x"].shape[-1]

    x = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_length, 3, dtype=torch.float32)
    coord_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    y = torch.zeros(batch_size, max_length, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        length = item["length"]
        x[batch_index, :length] = item["x"]
        coords[batch_index, :length] = item["coords"]
        coord_mask[batch_index, :length] = item["coord_mask"]
        y[batch_index, :length] = item["y"]
        valid_mask[batch_index, :length] = True

    return {
        "ids": [item["id"] for item in batch],
        "files": [item["file"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "x": x,
        "coords": coords,
        "coord_mask": coord_mask,
        "y": y,
        "mask": valid_mask,
    }


def has_feature_files(path: Path | None) -> bool:
    """返回目录是否存在且至少包含一个 ``.pt`` 文件。"""

    return path is not None and path.is_dir() and any(path.glob("*.pt"))
