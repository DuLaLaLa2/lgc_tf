"""并行分支共用的复现、指标和文件输出工具。"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(count: int, ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """确定性输出训练索引和验证索引。"""

    if count < 2:
        raise ValueError("至少需要两个样本才能划分训练/验证集")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_size = min(count - 1, max(1, int(count * ratio)))
    return indices[val_size:], indices[:val_size]


def binary_metrics(labels: Sequence, probabilities: Sequence, threshold: float) -> Dict[str, Any]:
    """输入标签、正类概率和阈值，输出残基级二分类指标字典。"""

    y_true = np.asarray(labels).astype(int)
    y_prob = np.asarray(probabilities).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    if np.unique(y_true).size == 2:
        auc = float(roc_auc_score(y_true, y_prob))
        aupr = float(average_precision_score(y_true, y_prob))
    else:
        auc = aupr = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "aupr": aupr,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def best_mcc_threshold(
    labels: Sequence, probabilities: Sequence, start: float, end: float, step: float
) -> Tuple[float, Dict[str, Any]]:
    """仅在验证数据上搜索最大 MCC 阈值。"""

    best_threshold = start
    best_metrics = None
    for threshold in np.arange(start, end + step / 2, step):
        metrics = binary_metrics(labels, probabilities, float(threshold))
        if best_metrics is None or metrics["mcc"] > best_metrics["mcc"]:
            best_threshold, best_metrics = float(threshold), metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    """返回模型总参数量和可训练参数量。"""

    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=True)


def save_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

