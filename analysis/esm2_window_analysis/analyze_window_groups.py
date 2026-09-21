"""按照真实 ESM2 滑窗数量重新聚合已有测试集逐残基预测。"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# 这些参数复制自原始特征提取脚本，不从测试结果反推。
WINDOW_SIZE = 1022
STRIDE = 900
OVERLAP = WINDOW_SIZE - STRIDE
EXTRACTION_SOURCE = (
    r"C:\Code\Git_clone_resp\3D-SRPE-main\3D-SRPE-main"
    r"\extract_esm2_from_rna_dataset.py"
)

ID_ALIASES = ("protein_id", "sample_id", "id")
LENGTH_ALIASES = ("sequence_length", "protein_length", "length")
LABEL_ALIASES = ("label", "y_true", "labels")
PROBABILITY_ALIASES = ("probability", "y_prob", "pred_prob", "probabilities")
RESIDUE_ALIASES = ("residue_index", "residue_id", "position")
THRESHOLD_ALIASES = ("threshold", "best_threshold")

OUTPUT_FILES = (
    "window_analysis_protein_metadata.csv",
    "protein_metadata.csv",
    "transformer_by_esm2_window.csv",
    "bimamba_by_esm2_window.csv",
    "transformer_vs_bimamba_by_window.csv",
    "per_protein_transformer_vs_bimamba.csv",
    "window_bootstrap_delta_mcc.csv",
    "correlation_analysis.csv",
    "plot_window_metrics_long.csv",
    "plot_sequence_length_delta_mcc.csv",
    "analysis_warnings.log",
    "scientific_summary.md",
)


def generate_window_ranges(seq_len: int) -> list[tuple[int, int]]:
    """逐句复现原提取脚本的 start/end/break/stride 逻辑。"""
    if seq_len <= 0:
        raise ValueError(f"sequence_length 必须为正整数，实际为 {seq_len}")

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(start + WINDOW_SIZE, seq_len)
        ranges.append((start, end))
        if end == seq_len:
            break
        start += STRIDE
    return ranges


def get_esm2_window_count(seq_len: int) -> int:
    """根据原始循环而非硬编码长度区间计算窗口数。"""
    return len(generate_window_ranges(int(seq_len)))


def run_boundary_tests() -> None:
    """验证任务指定的全部临界点。"""
    expected = {
        500: 1,
        1000: 1,
        1022: 1,
        1023: 2,
        1203: 2,
        1343: 2,
        1500: 2,
        1922: 2,
        1923: 3,
        2000: 3,
        2485: 3,
        2822: 3,
        2823: 4,
    }
    actual = {length: get_esm2_window_count(length) for length in expected}
    if actual != expected:
        raise AssertionError(f"滑窗边界测试失败：expected={expected}, actual={actual}")

    # 同时验证循环结果与闭式公式一致，但统计仍调用上面的真实循环。
    for length in range(1, 10001):
        formula = 1 if length <= WINDOW_SIZE else 1 + math.ceil((length - WINDOW_SIZE) / STRIDE)
        if get_esm2_window_count(length) != formula:
            raise AssertionError(f"滑窗公式与原循环不一致：L={length}")


def first_existing_column(frame: pd.DataFrame, aliases: Iterable[str], role: str) -> str:
    """从允许的别名中选择输入列，缺失时明确报错。"""
    for name in aliases:
        if name in frame.columns:
            return name
    raise ValueError(f"缺少 {role} 列；允许列名：{list(aliases)}")


def read_predictions(path: Path, method: str, explicit_threshold: float | None) -> tuple[pd.DataFrame, float]:
    """读取并规范化一份既有逐残基预测，不改变概率或标签。"""
    if not path.is_file():
        raise FileNotFoundError(f"{method} 预测文件不存在：{path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{method} 预测文件为空：{path}")

    id_col = first_existing_column(raw, ID_ALIASES, "protein_id")
    length_col = first_existing_column(raw, LENGTH_ALIASES, "sequence_length")
    label_col = first_existing_column(raw, LABEL_ALIASES, "label")
    probability_col = first_existing_column(raw, PROBABILITY_ALIASES, "probability")
    residue_col = first_existing_column(raw, RESIDUE_ALIASES, "residue_index")

    frame = raw[[id_col, length_col, residue_col, label_col, probability_col]].copy()
    frame.columns = ["protein_id", "sequence_length", "residue_index", "label", "probability"]
    frame["protein_id"] = frame["protein_id"].astype(str)
    for column in ("sequence_length", "residue_index", "label", "probability"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["sequence_length"] = frame["sequence_length"].astype(int)
    frame["residue_index"] = frame["residue_index"].astype(int)
    frame["label"] = frame["label"].astype(int)

    if frame[["protein_id", "residue_index"]].duplicated().any():
        duplicates = frame.loc[
            frame[["protein_id", "residue_index"]].duplicated(False),
            ["protein_id", "residue_index"],
        ].head(10)
        raise ValueError(f"{method} 存在重复残基键：\n{duplicates.to_string(index=False)}")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{method} label 不是严格二分类 0/1")
    if not np.isfinite(frame["probability"]).all() or not frame["probability"].between(0, 1).all():
        raise ValueError(f"{method} probability 含非有限值或超出 [0,1]")

    threshold_col = next((name for name in THRESHOLD_ALIASES if name in raw.columns), None)
    file_threshold: float | None = None
    if threshold_col is not None:
        values = pd.to_numeric(raw[threshold_col], errors="raise").dropna().unique()
        if len(values) != 1:
            raise ValueError(f"{method} 的 {threshold_col} 必须全文件唯一，实际为 {values.tolist()}")
        file_threshold = float(values[0])

    if explicit_threshold is None and file_threshold is None:
        raise ValueError(
            f"{method} 无 threshold 列；必须用命令行传入该模型由验证集确定的正式 threshold"
        )
    threshold = float(explicit_threshold if explicit_threshold is not None else file_threshold)
    if file_threshold is not None and explicit_threshold is not None and not np.isclose(
        file_threshold, explicit_threshold, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"{method} 显式 threshold={explicit_threshold} 与文件 threshold={file_threshold} 不一致"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{method} threshold 超出 [0,1]：{threshold}")

    frame = frame.sort_values(["protein_id", "residue_index"], kind="stable").reset_index(drop=True)
    validate_internal_lengths(frame, method)
    return frame, threshold


def validate_internal_lengths(frame: pd.DataFrame, method: str) -> None:
    """校验每个蛋白的声明长度、残基行数和残基索引连续性。"""
    for protein_id, group in frame.groupby("protein_id", sort=False):
        lengths = group["sequence_length"].unique()
        if len(lengths) != 1:
            raise ValueError(f"{method} protein_id={protein_id} 存在多个 sequence_length：{lengths}")
        declared = int(lengths[0])
        if len(group) != declared:
            raise ValueError(
                f"{method} protein_id={protein_id} 声明长度={declared}，逐残基行数={len(group)}"
            )
        indices = group["residue_index"].to_numpy()
        expected_zero = np.arange(declared)
        expected_one = np.arange(1, declared + 1)
        if not (np.array_equal(indices, expected_zero) or np.array_equal(indices, expected_one)):
            raise ValueError(f"{method} protein_id={protein_id} 的 residue_index 不连续")


def align_and_validate(transformer: pd.DataFrame, bimamba: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """显式验证同一批蛋白、长度、窗口数、残基索引和标签。"""
    transformer_ids = sorted(transformer["protein_id"].unique().tolist())
    bimamba_ids = sorted(bimamba["protein_id"].unique().tolist())
    if transformer_ids != bimamba_ids:
        only_t = sorted(set(transformer_ids) - set(bimamba_ids))
        only_m = sorted(set(bimamba_ids) - set(transformer_ids))
        raise AssertionError(f"protein_id 不一致；仅 Transformer={only_t}；仅 BiMamba={only_m}")

    key = ["protein_id", "residue_index"]
    t = transformer.sort_values(key, kind="stable").reset_index(drop=True)
    m = bimamba.sort_values(key, kind="stable").reset_index(drop=True)
    if not t[key].equals(m[key]):
        mismatch = pd.concat([t[key].add_prefix("transformer_"), m[key].add_prefix("bimamba_")], axis=1)
        raise AssertionError(f"逐残基键不一致：\n{mismatch.loc[(t[key] != m[key]).any(axis=1)].head(10)}")

    for protein_id in transformer_ids:
        t_group = t[t["protein_id"] == protein_id]
        m_group = m[m["protein_id"] == protein_id]
        t_length = int(t_group["sequence_length"].iloc[0])
        m_length = int(m_group["sequence_length"].iloc[0])
        if t_length != m_length:
            raise AssertionError(
                f"protein_id={protein_id} sequence_length 不一致：Transformer={t_length}, BiMamba={m_length}"
            )
        t_windows = get_esm2_window_count(t_length)
        m_windows = get_esm2_window_count(m_length)
        if t_windows != m_windows:
            raise AssertionError(
                f"protein_id={protein_id} window_count 不一致：Transformer={t_windows}, BiMamba={m_windows}"
            )
        if not np.array_equal(t_group["label"].to_numpy(), m_group["label"].to_numpy()):
            raise AssertionError(f"protein_id={protein_id} labels 不一致")
    return t, m


def window_group(window_count: int) -> str:
    """前三个窗口数单列，四个及以上合并且不生成空组。"""
    return f"{window_count}-window" if window_count <= 3 else ">=4-window"


def safe_divide(numerator: float, denominator: float) -> float:
    """仅用于有明确定义的计数比率。"""
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    """计算与正式测试一致的 residue pooling 指标。"""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    two_classes = np.unique(labels).size == 2
    return {
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)) if two_classes else float("nan"),
        "aupr": float(average_precision_score(labels, probabilities)) if two_classes else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def compute_per_protein(
    frame: pd.DataFrame,
    threshold: float,
    prefix: str,
    warning_messages: list[str],
) -> pd.DataFrame:
    """计算逐蛋白指标；单类别蛋白的 MCC/AUC/AUPR 保留 NaN。"""
    rows: list[dict[str, float | int | str]] = []
    for protein_id, group in frame.groupby("protein_id", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        probabilities = group["probability"].to_numpy(dtype=float)
        metrics = compute_metrics(labels, probabilities, threshold)
        if np.unique(labels).size < 2:
            metrics["mcc"] = float("nan")
            metrics["auc"] = float("nan")
            metrics["aupr"] = float("nan")
            warning_messages.append(
                f"{prefix}: protein_id={protein_id} 真实标签只有一个类别；MCC/AUC/AUPR 保存为 NaN"
            )
        rows.append(
            {
                "protein_id": protein_id,
                f"{prefix}_mcc": metrics["mcc"],
                f"{prefix}_f1": metrics["f1"],
                f"{prefix}_precision": metrics["precision"],
                f"{prefix}_recall": metrics["recall"],
                f"{prefix}_specificity": metrics["specificity"],
                f"{prefix}_accuracy": metrics["accuracy"],
                f"{prefix}_auc": metrics["auc"],
                f"{prefix}_aupr": metrics["aupr"],
            }
        )
    return pd.DataFrame(rows)


def build_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """由共同真实标签建立唯一的蛋白元数据表。"""
    rows: list[dict[str, float | int | str]] = []
    for protein_id, group in frame.groupby("protein_id", sort=True):
        length = int(group["sequence_length"].iloc[0])
        positive = int(group["label"].sum())
        negative = length - positive
        count = get_esm2_window_count(length)
        rows.append(
            {
                "protein_id": protein_id,
                "sequence_length": length,
                "esm2_window_count": count,
                "window_group": window_group(count),
                "positive_residue_count": positive,
                "negative_residue_count": negative,
                "positive_ratio": positive / length,
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    per_protein: pd.DataFrame,
    threshold: float,
    method_name: str,
    prefix: str,
) -> pd.DataFrame:
    """同时生成 residue-level micro 与 protein-level macro 指标。"""
    tagged = frame.merge(
        metadata[["protein_id", "window_group", "esm2_window_count"]],
        on="protein_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, float | int | str]] = []
    ordered_groups = metadata.sort_values("esm2_window_count")["window_group"].drop_duplicates().tolist()
    for group_name in ordered_groups:
        selected = tagged[tagged["window_group"] == group_name]
        proteins = metadata[metadata["window_group"] == group_name]
        metrics = compute_metrics(
            selected["label"].to_numpy(), selected["probability"].to_numpy(), threshold
        )
        protein_metrics = per_protein[per_protein["protein_id"].isin(proteins["protein_id"])]
        positive_count = int(selected["label"].sum())
        residue_count = len(selected)
        rows.append(
            {
                "method": method_name,
                "window_group": group_name,
                "protein_count": len(proteins),
                "residue_count": residue_count,
                "positive_count": positive_count,
                "negative_count": residue_count - positive_count,
                "positive_ratio": positive_count / residue_count,
                **metrics,
                "micro_mcc": metrics["mcc"],
                "macro_mcc": protein_metrics[f"{prefix}_mcc"].mean(skipna=True),
                "micro_f1": metrics["f1"],
                "macro_f1": protein_metrics[f"{prefix}_f1"].mean(skipna=True),
                "micro_aupr": metrics["aupr"],
                "macro_aupr": protein_metrics[f"{prefix}_aupr"].mean(skipna=True),
                "macro_valid_mcc_protein_count": int(protein_metrics[f"{prefix}_mcc"].notna().sum()),
                "macro_valid_aupr_protein_count": int(protein_metrics[f"{prefix}_aupr"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_per_protein_comparison(
    metadata: pd.DataFrame,
    transformer_metrics: pd.DataFrame,
    bimamba_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """合并两种模型逐蛋白结果并计算 BiMamba - Transformer。"""
    result = metadata.rename(columns={"positive_residue_count": "positive_count"}).merge(
        transformer_metrics, on="protein_id", validate="one_to_one"
    ).merge(bimamba_metrics, on="protein_id", validate="one_to_one")
    for metric in ("mcc", "f1", "precision", "recall", "specificity", "accuracy", "auc", "aupr"):
        result[f"delta_{metric}"] = result[f"bimamba_{metric}"] - result[f"transformer_{metric}"]
    columns = [
        "protein_id",
        "sequence_length",
        "esm2_window_count",
        "window_group",
        "positive_count",
        "positive_ratio",
        "transformer_mcc",
        "bimamba_mcc",
        "delta_mcc",
        "transformer_f1",
        "bimamba_f1",
        "delta_f1",
        "transformer_aupr",
        "bimamba_aupr",
        "delta_aupr",
        "transformer_precision",
        "bimamba_precision",
        "delta_precision",
        "transformer_recall",
        "bimamba_recall",
        "delta_recall",
        "transformer_auc",
        "bimamba_auc",
        "delta_auc",
        "transformer_specificity",
        "bimamba_specificity",
        "delta_specificity",
        "transformer_accuracy",
        "bimamba_accuracy",
        "delta_accuracy",
    ]
    return result[columns]


def build_group_comparison(transformer: pd.DataFrame, bimamba: pd.DataFrame) -> pd.DataFrame:
    """构造窗口组核心对比表。"""
    shared = ["window_group", "protein_count", "residue_count", "positive_ratio"]
    metrics = ("mcc", "aupr", "f1", "precision", "recall", "auc", "specificity", "accuracy")
    t_columns = shared + [item for metric in metrics for item in (metric, f"macro_{metric}") if item in transformer]
    m_columns = shared + [item for metric in metrics for item in (metric, f"macro_{metric}") if item in bimamba]
    t = transformer[t_columns].rename(
        columns={name: f"transformer_{name}" for name in t_columns if name not in shared}
    )
    m = bimamba[m_columns].rename(
        columns={name: f"bimamba_{name}" for name in m_columns if name not in shared}
    )
    merged = t.merge(m, on=shared, validate="one_to_one")
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[f"bimamba_{metric}"] - merged[f"transformer_{metric}"]
        macro_t = f"transformer_macro_{metric}"
        macro_m = f"bimamba_macro_{metric}"
        if macro_t in merged and macro_m in merged:
            merged[f"delta_macro_{metric}"] = merged[macro_m] - merged[macro_t]
    return merged


def mcc_from_counts(tn: np.ndarray, fp: np.ndarray, fn: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """利用聚合混淆矩阵快速计算 bootstrap MCC，零分母与 sklearn 一致返回 0。"""
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def protein_confusion_counts(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """预先计算每条蛋白的混淆矩阵，避免 bootstrap 重复展开全部残基。"""
    rows: list[dict[str, int | str]] = []
    for protein_id, group in frame.groupby("protein_id", sort=True):
        predictions = (group["probability"].to_numpy() >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(group["label"], predictions, labels=[0, 1]).ravel()
        rows.append({"protein_id": protein_id, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return pd.DataFrame(rows)


def bootstrap_delta_mcc(
    metadata: pd.DataFrame,
    transformer: pd.DataFrame,
    bimamba: pd.DataFrame,
    transformer_threshold: float,
    bimamba_threshold: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """以蛋白为簇有放回采样，计算 pooled MCC 差值的百分位区间。"""
    t_counts = protein_confusion_counts(transformer, transformer_threshold).set_index("protein_id")
    m_counts = protein_confusion_counts(bimamba, bimamba_threshold).set_index("protein_id")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    ordered_groups = metadata.sort_values("esm2_window_count")["window_group"].drop_duplicates().tolist()
    for group_name in ordered_groups:
        ids = metadata.loc[metadata["window_group"] == group_name, "protein_id"].tolist()
        t = t_counts.loc[ids, ["tn", "fp", "fn", "tp"]].to_numpy(dtype=np.int64)
        m = m_counts.loc[ids, ["tn", "fp", "fn", "tp"]].to_numpy(dtype=np.int64)
        draws = rng.integers(0, len(ids), size=(n_bootstrap, len(ids)))
        t_sum = t[draws].sum(axis=1)
        m_sum = m[draws].sum(axis=1)
        t_mcc = mcc_from_counts(t_sum[:, 0], t_sum[:, 1], t_sum[:, 2], t_sum[:, 3])
        m_mcc = mcc_from_counts(m_sum[:, 0], m_sum[:, 1], m_sum[:, 2], m_sum[:, 3])
        delta = m_mcc - t_mcc
        observed_t = t.sum(axis=0)
        observed_m = m.sum(axis=0)
        observed = float(
            mcc_from_counts(*[np.asarray([value]) for value in observed_m])[0]
            - mcc_from_counts(*[np.asarray([value]) for value in observed_t])[0]
        )
        rows.append(
            {
                "window_group": group_name,
                "protein_count": len(ids),
                "observed_delta_mcc": observed,
                "bootstrap_mean_delta_mcc": float(np.mean(delta)),
                "ci_2.5": float(np.percentile(delta, 2.5)),
                "ci_97.5": float(np.percentile(delta, 97.5)),
                "n_bootstrap": n_bootstrap,
                "bootstrap_unit": "protein",
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def correlation_analysis(per_protein: pd.DataFrame) -> pd.DataFrame:
    """只对 delta_mcc 有定义的蛋白计算三项 Spearman 相关。"""
    rows: list[dict[str, float | int | str]] = []
    valid = per_protein.dropna(subset=["delta_mcc"])
    for variable in ("esm2_window_count", "sequence_length", "positive_ratio"):
        selected = valid[[variable, "delta_mcc"]].dropna()
        if len(selected) < 2 or selected[variable].nunique() < 2 or selected["delta_mcc"].nunique() < 2:
            rho, p_value = float("nan"), float("nan")
        else:
            rho, p_value = spearmanr(selected[variable], selected["delta_mcc"])
        rows.append(
            {
                "variable": variable,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
                "valid_protein_count": len(selected),
            }
        )
    return pd.DataFrame(rows)


def build_plot_window_data(transformer: pd.DataFrame, bimamba: pd.DataFrame) -> pd.DataFrame:
    """整理按窗口绘图所需的长表，不执行复杂绘图。"""
    rows: list[dict[str, float | str]] = []
    for source, method in ((transformer, "Transformer"), (bimamba, "BiMamba")):
        for row in source.itertuples(index=False):
            for metric in ("mcc", "aupr", "f1", "precision", "recall"):
                rows.append(
                    {
                        "window_group": row.window_group,
                        "method": method,
                        "metric": metric,
                        "aggregation": "micro",
                        "value": getattr(row, metric),
                    }
                )
            for metric in ("mcc", "aupr", "f1"):
                rows.append(
                    {
                        "window_group": row.window_group,
                        "method": method,
                        "metric": metric,
                        "aggregation": "macro",
                        "value": getattr(row, f"macro_{metric}"),
                    }
                )
        for row in source.itertuples(index=False):
            rows.append(
                {
                    "window_group": row.window_group,
                    "method": "Ground truth",
                    "metric": "positive_ratio",
                    "aggregation": "micro",
                    "value": row.positive_ratio,
                }
            )
        break
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """统一保存 CSV，并显式保留 NaN。"""
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.10g")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """在不引入 tabulate 额外依赖的情况下生成简单 Markdown 表。"""
    columns = [str(column) for column in frame.columns]

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "NaN"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def ensure_no_overwrite(output_dir: Path) -> None:
    """遵守不覆盖现有结果文件的要求。"""
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing:
        paths = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"以下结果已存在，拒绝覆盖；请指定新的 --output-dir：\n{paths}")


def print_core_tables(comparison: pd.DataFrame, transformer: pd.DataFrame) -> None:
    """按任务要求在终端打印简洁核心表。"""
    print("=" * 80)
    print("Transformer vs BiMamba by ESM2 Window Count")
    print(
        comparison[
            ["window_group", "protein_count", "residue_count", "transformer_mcc", "bimamba_mcc", "delta_mcc"]
        ].to_string(index=False)
    )
    print("\nWindow AUPR")
    print(
        comparison[
            ["window_group", "transformer_aupr", "bimamba_aupr", "delta_aupr"]
        ].to_string(index=False)
    )
    print("\nClass balance")
    print(
        transformer[
            ["window_group", "protein_count", "residue_count", "positive_count", "positive_ratio"]
        ].to_string(index=False)
    )
    print("=" * 80)


def write_scientific_summary(
    path: Path,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    correlations: pd.DataFrame,
    transformer_threshold: float,
    bimamba_threshold: float,
) -> None:
    """输出谨慎的描述性结论，避免把相关性写成跨窗口恢复的因果证明。"""
    lines = [
        "# ESM2 window-count post-hoc analysis",
        "",
        "本分析仅重新聚合既有测试集逐残基预测，没有重新训练、重新提取 ESM2、改变测试集或重新选择阈值。",
        "",
        f"- Transformer validation-derived threshold: {transformer_threshold:g}",
        f"- BiMamba validation-derived threshold: {bimamba_threshold:g}",
        f"- Sliding window: window_size={WINDOW_SIZE}, overlap={OVERLAP}, stride={STRIDE}",
        "",
        "## 核心结果",
        "",
        dataframe_to_markdown(comparison),
        "",
        "## Protein-level bootstrap（ΔMCC）",
        "",
        dataframe_to_markdown(bootstrap),
        "",
        "## Spearman 相关",
        "",
        dataframe_to_markdown(correlations),
        "",
        "## 科学解释边界",
        "",
    ]
    delta_by_group = comparison.set_index("window_group")["delta_mcc"].to_dict()
    if "1-window" in delta_by_group:
        lines.append(f"- 1-window 的 ΔMCC={delta_by_group['1-window']:.6f}。")
    if "2-window" in delta_by_group:
        lines.append(f"- 2-window 的 ΔMCC={delta_by_group['2-window']:.6f}。")
    if "3-window" in delta_by_group:
        lines.append(f"- 3-window 的 ΔMCC={delta_by_group['3-window']:.6f}。")
    lines.extend(
        [
            "- 是否随 window_count 系统变化，应同时结合 macro 指标、蛋白级 bootstrap 区间和 Spearman 结果判断。",
            "- positive_ratio 若随 window_count 改变，则类别不平衡与窗口数量存在混杂，不能把 pooled 指标变化单独归因于跨窗口上下文。",
            "- 即使多窗口组增益更大，这一 post-hoc 结果也只能支持关联性证据，不能自动证明 BiMamba 已恢复 ESM2 跨窗口信息。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析输入路径、正式阈值和 bootstrap 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-predictions", type=Path, default="/lgc_tf/src/outputs/predictions/baseline_test_predictions.csv")
    parser.add_argument("--bimamba-predictions", type=Path, default="/lgc_tf/BiMamba_experiments/esm2_lgc_bimamba/results/test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--transformer-threshold", type=float)
    parser.add_argument("--bimamba-threshold", type=float)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """执行完整的只读预测、重新分组统计流程。"""
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap 必须为正整数")
    run_boundary_tests()
    print("滑窗边界测试：通过")
    print(f"原 ESM2 滑窗代码：{EXTRACTION_SOURCE}")
    print(f"window_size={WINDOW_SIZE}, overlap={OVERLAP}, stride={STRIDE}")

    output_dir = args.output_dir.resolve()
    ensure_no_overwrite(output_dir)
    transformer, transformer_threshold = read_predictions(
        args.transformer_predictions.resolve(), "Transformer", args.transformer_threshold
    )
    bimamba, bimamba_threshold = read_predictions(
        args.bimamba_predictions.resolve(), "BiMamba", args.bimamba_threshold
    )
    transformer, bimamba = align_and_validate(transformer, bimamba)
    metadata = build_metadata(transformer)

    print(f"Transformer 预测结果来源：{args.transformer_predictions.resolve()}")
    print(f"BiMamba 预测结果来源：{args.bimamba_predictions.resolve()}")
    print(f"Transformer threshold：{transformer_threshold:g}")
    print(f"BiMamba threshold：{bimamba_threshold:g}")
    print(f"测试蛋白总数：{len(metadata)}")
    print("是否需要重新 inference：否")
    print("是否完全不需要重新训练：是")

    warning_messages: list[str] = []
    transformer_per = compute_per_protein(
        transformer, transformer_threshold, "transformer", warning_messages
    )
    bimamba_per = compute_per_protein(bimamba, bimamba_threshold, "bimamba", warning_messages)
    per_protein = build_per_protein_comparison(metadata, transformer_per, bimamba_per)
    transformer_groups = summarize_groups(
        transformer,
        metadata,
        transformer_per,
        transformer_threshold,
        "ESM2 + LGC + Transformer",
        "transformer",
    )
    bimamba_groups = summarize_groups(
        bimamba,
        metadata,
        bimamba_per,
        bimamba_threshold,
        "ESM2 + LGC + BiMamba",
        "bimamba",
    )
    comparison = build_group_comparison(transformer_groups, bimamba_groups)
    bootstrap = bootstrap_delta_mcc(
        metadata,
        transformer,
        bimamba,
        transformer_threshold,
        bimamba_threshold,
        args.n_bootstrap,
        args.seed,
    )
    correlations = correlation_analysis(per_protein)
    plot_window = build_plot_window_data(transformer_groups, bimamba_groups)
    plot_sequence = per_protein[
        ["protein_id", "sequence_length", "esm2_window_count", "window_group", "positive_ratio", "delta_mcc"]
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metadata, output_dir / "window_analysis_protein_metadata.csv")
    write_csv(metadata, output_dir / "protein_metadata.csv")
    write_csv(transformer_groups, output_dir / "transformer_by_esm2_window.csv")
    write_csv(bimamba_groups, output_dir / "bimamba_by_esm2_window.csv")
    write_csv(comparison, output_dir / "transformer_vs_bimamba_by_window.csv")
    write_csv(per_protein, output_dir / "per_protein_transformer_vs_bimamba.csv")
    write_csv(bootstrap, output_dir / "window_bootstrap_delta_mcc.csv")
    write_csv(correlations, output_dir / "correlation_analysis.csv")
    write_csv(plot_window, output_dir / "plot_window_metrics_long.csv")
    write_csv(plot_sequence, output_dir / "plot_sequence_length_delta_mcc.csv")
    (output_dir / "analysis_warnings.log").write_text(
        "\n".join(warning_messages) + ("\n" if warning_messages else ""), encoding="utf-8"
    )
    write_scientific_summary(
        output_dir / "scientific_summary.md",
        comparison,
        bootstrap,
        correlations,
        transformer_threshold,
        bimamba_threshold,
    )
    for message in warning_messages:
        warnings.warn(message, RuntimeWarning)
    print_core_tables(comparison, transformer_groups)
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()
