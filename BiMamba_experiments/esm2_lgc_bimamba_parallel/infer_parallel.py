"""并行分支测试入口：只加载权重、预测测试集并保存统计结果。"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from config import DATA, INFERENCE, OUTPUT
from dataset_adapter import FullLengthRNADataset, pad_collate_full_length
from model_parallel import ESM2ParallelLGCBiMamba
from src.models.lgc_transformer_model import ESM2LGCTransformer
from utils_parallel import binary_metrics, save_csv, save_json


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """加载 checkpoint；路径不存在时立即报错。"""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_baseline(checkpoint: Dict[str, Any]) -> ESM2LGCTransformer:
    """根据已有 Transformer checkpoint 配置重建基线模型。"""

    config = checkpoint.get("config", {})
    return ESM2LGCTransformer(
        input_dim=int(config.get("input_dim", 2560)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        ffn_dim=int(config.get("ffn_dim", 2048)),
        dropout=float(config.get("dropout", 0.1)),
        sde_dim=int(config.get("sde_dim", 16)),
        cutoff=float(config.get("cutoff", 30.0)),
        k_neighbors=int(config.get("k_neighbors", 16)),
    )


def model_forward(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    parallel_model: bool,
) -> torch.Tensor:
    """输入单蛋白 batch，输出 logits[1,L]。"""

    inputs = {
        "x": batch["x"].to(device, non_blocking=True),
        "coords": batch["coords"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "coord_mask": batch["coord_mask"].to(device, non_blocking=True),
    }
    if parallel_model:
        inputs["lengths"] = batch["lengths"].to(device, non_blocking=True)
    return model(**inputs)


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    threshold: float,
    method: str,
    device: torch.device,
    amp_enabled: bool,
    parallel_model: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """按固定验证阈值预测测试集。

    返回逐残基预测、逐蛋白运行时间/显存和整体测试指标。
    """

    model.eval()
    prediction_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    probabilities: List[float] = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch["ids"]) != 1:
                raise ValueError("运行时间统计要求 batch_size=1")
            sample_id = batch["ids"][0]
            length = int(batch["lengths"][0].item())
            before_memory = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                before_memory = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            try:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model_forward(model, batch, device, parallel_model)
            except torch.cuda.OutOfMemoryError as error:
                memory = torch.cuda.memory_allocated(device) / 1024**2
                raise RuntimeError(
                    f"测试 OOM：sample_id={sample_id}, length={length}, GPU={memory:.1f}MB；未截断。"
                ) from error
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak_memory = (
                max(torch.cuda.max_memory_allocated(device) - before_memory, 0) / 1024**2
                if device.type == "cuda"
                else 0.0
            )
            sample_probabilities = torch.sigmoid(logits[0, :length]).float().cpu().tolist()
            sample_labels = batch["y"][0, :length].int().tolist()
            for residue_index, (label, probability) in enumerate(
                zip(sample_labels, sample_probabilities), start=1
            ):
                labels.append(label)
                probabilities.append(probability)
                prediction_rows.append(
                    {
                        "method": method,
                        "sample_id": sample_id,
                        "file": batch["files"][0],
                        "protein_length": length,
                        "residue_index": residue_index,
                        "label": label,
                        "probability": probability,
                        "prediction": int(probability >= threshold),
                        "threshold": threshold,
                    }
                )
            runtime_rows.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "protein_length": length,
                    "inference_time_seconds": elapsed,
                    "peak_gpu_memory_mb": peak_memory,
                }
            )
    metrics = binary_metrics(labels, probabilities, threshold)
    metrics.update(protein_count=len(runtime_rows), residue_count=len(labels))
    return prediction_rows, runtime_rows, metrics


def length_group_rows(
    rows: List[Dict[str, Any]], method: str, threshold: float
) -> List[Dict[str, Any]]:
    """将测试蛋白按长度分组，返回各组残基级指标。"""

    groups = (
        ("<500", 0, 500),
        ("500-1000", 500, 1000),
        ("1000-1500", 1000, 1500),
        ("1500-2000", 1500, 2000),
        (">=2000", 2000, float("inf")),
    )
    results = []
    for name, lower, upper in groups:
        selected = [row for row in rows if lower <= row["protein_length"] < upper]
        if not selected:
            continue
        metrics = binary_metrics(
            [row["label"] for row in selected],
            [row["probability"] for row in selected],
            threshold,
        )
        results.append(
            {
                "method": method,
                "length_group": name,
                "protein_count": len({row["sample_id"] for row in selected}),
                "residue_count": len(selected),
                **metrics,
            }
        )
    return results


def make_summary(
    method: str, metrics: Dict[str, Any], checkpoint: Dict[str, Any]
) -> Dict[str, Any]:
    """合并 checkpoint 配置和测试指标，形成 summary.csv 一行。"""

    config = checkpoint.get("model_config", checkpoint.get("config", {}))
    return {
        "method": method,
        "fusion": checkpoint.get("fusion", ""),
        "val_mcc": checkpoint.get("best_val_mcc"),
        "test_mcc": metrics["mcc"],
        "test_auc": metrics["auc"],
        "test_aupr": metrics["aupr"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_specificity": metrics["specificity"],
        "best_threshold": checkpoint.get("best_threshold"),
        "best_epoch": checkpoint.get("epoch"),
        "d_model": config.get("d_model", config.get("hidden_dim")),
        "num_bimamba_layers": config.get("num_bimamba_layers", ""),
        "d_state": config.get("d_state", ""),
        "d_conv": config.get("d_conv", ""),
        "expand": config.get("expand", ""),
        "dropout": config.get("dropout"),
        "seed": checkpoint.get("training_config", {}).get("seed", 42),
    }


def main() -> None:
    """加载最佳并行模型，以验证阈值完成测试预测和分析。"""

    OUTPUT.result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = INFERENCE.use_amp and device.type == "cuda"
    dataset = FullLengthRNADataset(DATA.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=INFERENCE.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_collate_full_length,
    )

    checkpoint_path = OUTPUT.checkpoint_dir / OUTPUT.best_checkpoint_name
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = ESM2ParallelLGCBiMamba(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint["best_threshold"])
    method = "ESM2 + Parallel LGC/BiMamba + Concat"
    predictions, runtime_rows, metrics = predict(
        model, loader, threshold, method, device, amp_enabled, parallel_model=True
    )

    prediction_fields = [
        "method",
        "sample_id",
        "file",
        "protein_length",
        "residue_index",
        "label",
        "probability",
        "prediction",
        "threshold",
    ]
    save_csv(OUTPUT.result_dir / "test_predictions.csv", predictions, prediction_fields)
    group_rows = length_group_rows(predictions, method, threshold)
    summaries = [make_summary(method, metrics, checkpoint)]
    all_metrics: Dict[str, Any] = {method: metrics}

    if INFERENCE.baseline_checkpoint is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        baseline_checkpoint = load_checkpoint(INFERENCE.baseline_checkpoint, device)
        baseline_model = build_baseline(baseline_checkpoint).to(device)
        baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
        baseline_threshold = float(baseline_checkpoint["best_threshold"])
        baseline_method = "ESM2 + LGC + Transformer"
        baseline_predictions, baseline_runtime, baseline_metrics = predict(
            baseline_model,
            loader,
            baseline_threshold,
            baseline_method,
            device,
            amp_enabled,
            parallel_model=False,
        )
        save_csv(
            OUTPUT.result_dir / "baseline_test_predictions.csv",
            baseline_predictions,
            prediction_fields,
        )
        runtime_rows.extend(baseline_runtime)
        group_rows.extend(
            length_group_rows(baseline_predictions, baseline_method, baseline_threshold)
        )
        summaries.append(make_summary(baseline_method, baseline_metrics, baseline_checkpoint))
        all_metrics[baseline_method] = baseline_metrics
        save_csv(
            OUTPUT.result_dir / "comparison_deltas.csv",
            [
                {
                    "comparison": "Parallel - Transformer",
                    **{
                        f"delta_{name}": metrics[name] - baseline_metrics[name]
                        for name in ("mcc", "aupr", "auc", "f1")
                    },
                }
            ],
            ["comparison", "delta_mcc", "delta_aupr", "delta_auc", "delta_f1"],
        )

    save_csv(
        OUTPUT.result_dir / "runtime_by_length.csv",
        runtime_rows,
        ["method", "sample_id", "protein_length", "inference_time_seconds", "peak_gpu_memory_mb"],
    )
    save_csv(
        OUTPUT.result_dir / "length_group_metrics.csv",
        group_rows,
        [
            "method",
            "length_group",
            "protein_count",
            "residue_count",
            "mcc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "auc",
            "aupr",
            "accuracy",
            "tp",
            "tn",
            "fp",
            "fn",
            "threshold",
        ],
    )
    save_csv(
        OUTPUT.result_dir / "summary.csv",
        summaries,
        [
            "method",
            "fusion",
            "val_mcc",
            "test_mcc",
            "test_auc",
            "test_aupr",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_specificity",
            "best_threshold",
            "best_epoch",
            "d_model",
            "num_bimamba_layers",
            "d_state",
            "d_conv",
            "expand",
            "dropout",
            "seed",
        ],
    )
    save_json(OUTPUT.result_dir / "test_metrics.json", all_metrics)
    print(f"测试完成：threshold={threshold:.2f}, MCC={metrics['mcc']:.4f}, AUPR={metrics['aupr']:.4f}")
    print(f"结果目录：{OUTPUT.result_dir}")


if __name__ == "__main__":
    main()

