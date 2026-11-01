"""训练和推理共享的小型工具函数。"""

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


def set_reproducible_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机状态。输入为整数种子，无输出。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(sample_count: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """确定性划分样本索引，输出 ``(train_indices, val_indices)``。"""

    if sample_count < 2:
        raise ValueError("训练目录至少需要两个蛋白，才能划分训练集和验证集")
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    val_size = min(sample_count - 1, max(1, int(sample_count * val_ratio)))
    return indices[val_size:], indices[:val_size]


def compute_binary_metrics(y_true: Sequence, y_prob: Sequence, threshold: float) -> Dict[str, Any]:
    """计算残基级二分类指标。

    输入：等长标签、概率序列及固定阈值。
    输出：包含 AUC、AUPR、MCC、F1、Precision、Recall、Specificity 等的字典。
    """

    labels = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob).astype(float)
    predictions = (probabilities >= threshold).astype(int)
    metrics: Dict[str, Any] = {}
    if np.unique(labels).size == 2:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
        metrics["aupr"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["auc"] = float("nan")
        metrics["aupr"] = float("nan")

    metrics["mcc"] = float(matthews_corrcoef(labels, predictions))
    metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))
    metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    metrics.update(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), threshold=float(threshold))
    return metrics


def find_best_mcc_threshold(
    y_true: Sequence,
    y_prob: Sequence,
    start: float,
    end: float,
    step: float,
) -> Tuple[float, Dict[str, Any]]:
    """在验证集搜索 MCC 最大的阈值，输出 ``(阈值, 指标字典)``。"""

    best_threshold = start
    best_metrics: Dict[str, Any] | None = None
    for threshold in np.arange(start, end + step / 2.0, step):
        metrics = compute_binary_metrics(y_true, y_prob, float(threshold))
        if best_metrics is None or metrics["mcc"] > best_metrics["mcc"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """输出 ``(总参数量, 可训练参数量)``。"""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def save_json(path: Path, payload: Any) -> None:
    """以 UTF-8 JSON 保存可序列化对象。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=True)


def save_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    """保存字典行列表；即使没有数据行也会写入表头。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

