"""测试集推理入口：加载训练产物，预测测试集并保存分析结果。

本文件不训练、不重新选择阈值。修改 ``config.py`` 后可在 IDE 中直接运行。
"""

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
from model_bimamba import ESM2LGCBiMamba
from src.models.lgc_transformer_model import ESM2LGCTransformer
from utils_bimamba import compute_binary_metrics, save_csv, save_json


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """加载 checkpoint 字典；输入路径与设备，输出完整 checkpoint。"""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    try:
        print(f"加载 checkpoint：{path}")
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_baseline_model(checkpoint: Dict[str, Any]) -> ESM2LGCTransformer:
    """从现有基线 checkpoint 配置重建 Transformer 模型。"""

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


def run_model(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    is_bimamba: bool,
) -> torch.Tensor:
    """把一个单蛋白 batch 输入指定模型，输出 logits[1,L]。"""

    model_inputs = {
        "x": batch["x"].to(device, non_blocking=True),
        "coords": batch["coords"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "coord_mask": batch["coord_mask"].to(device, non_blocking=True),
    }
    if is_bimamba:
        model_inputs["lengths"] = batch["lengths"].to(device, non_blocking=True)
    return model(**model_inputs)


def predict_test_set(
    model: torch.nn.Module,
    loader: DataLoader,
    threshold: float,
    method: str,
    device: torch.device,
    amp_enabled: bool,
    is_bimamba: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """执行固定阈值测试推理。

    输出：逐残基预测行、逐蛋白运行时间行、整个测试集指标。运行时间要求 loader batch=1。
    """

    model.eval()
    prediction_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    all_labels: List[int] = []
    all_probabilities: List[float] = []

    with torch.inference_mode():
        for batch in loader:
            if len(batch["ids"]) != 1:
                raise ValueError("运行时间统计要求推理 batch_size=1")
            sample_id = batch["ids"][0]
            length = int(batch["lengths"][0].item())
            before_memory = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                before_memory = torch.cuda.memory_allocated(device)
            start_time = time.perf_counter()
            try:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = run_model(model, batch, device, is_bimamba)
            except torch.cuda.OutOfMemoryError as error:
                current_memory = torch.cuda.memory_allocated(device) / 1024**2
                raise RuntimeError(
                    f"测试样本发生 OOM：sample_id={sample_id}, length={length}, "
                    f"allocated={current_memory:.1f}MB；未截断或跳过该样本。"
                ) from error
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
            peak_memory_mb = (
                max(torch.cuda.max_memory_allocated(device) - before_memory, 0) / 1024**2
                if device.type == "cuda"
                else 0.0
            )

            probabilities = torch.sigmoid(logits[0, :length]).float().cpu().numpy()
            labels = batch["y"][0, :length].numpy().astype(int)
            for residue_index, (label, probability) in enumerate(zip(labels, probabilities), start=1):
                all_labels.append(int(label))
                all_probabilities.append(float(probability))
                prediction_rows.append(
                    {
                        "method": method,
                        "sample_id": sample_id,
                        "file": batch["files"][0],
                        "protein_length": length,
                        "residue_index": residue_index,
                        "label": int(label),
                        "probability": float(probability),
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
                    "peak_gpu_memory_mb": peak_memory_mb,
                }
            )

    metrics = compute_binary_metrics(all_labels, all_probabilities, threshold)
    metrics["protein_count"] = len(runtime_rows)
    metrics["residue_count"] = len(all_labels)
    return prediction_rows, runtime_rows, metrics


def compute_length_group_metrics(
    prediction_rows: List[Dict[str, Any]], method: str, threshold: float
) -> List[Dict[str, Any]]:
    """按蛋白长度分组后汇总残基级指标，输出可直接写 CSV 的字典行。"""

    groups = [
        ("<500", 0, 500),
        ("500-1000", 500, 1000),
        ("1000-1500", 1000, 1500),
        (">=1500", 1500, float("inf")),
        # ("1500-2000", 1500, 2000),
        # (">=2000", 2000, float("inf")),
    ]
    output_rows = []
    for group_name, lower, upper in groups:
        selected = [
            row for row in prediction_rows if lower <= row["protein_length"] < upper
        ]
        if not selected:
            continue
        metrics = compute_binary_metrics(
            [row["label"] for row in selected],
            [row["probability"] for row in selected],
            threshold,
        )
        protein_count = len({row["sample_id"] for row in selected})
        output_rows.append(
            {
                "method": method,
                "length_group": group_name,
                "protein_count": protein_count,
                "residue_count": len(selected),
                **metrics,
            }
        )
    return output_rows


def summary_row(
    method: str,
    metrics: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """把测试指标与 checkpoint 配置合并成 summary.csv 的一行。"""

    config = checkpoint.get("model_config", checkpoint.get("config", {}))
    return {
        "method": method,
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
    """加载最佳权重，以验证集阈值执行测试并保存所有结果。"""

    OUTPUT.result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = INFERENCE.use_amp and device.type == "cuda"
    test_dataset = FullLengthRNADataset(DATA.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=INFERENCE.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_collate_full_length,
    )

    branch_checkpoint_path = OUTPUT.checkpoint_dir / OUTPUT.best_checkpoint_name
    branch_checkpoint = load_checkpoint(branch_checkpoint_path, device)
    branch_model = ESM2LGCBiMamba(**branch_checkpoint["model_config"]).to(device)
    branch_model.load_state_dict(branch_checkpoint["model_state_dict"])
    threshold = float(branch_checkpoint["best_threshold"])
    branch_predictions, runtime_rows, branch_metrics = predict_test_set(
        branch_model,
        test_loader,
        threshold,
        "ESM2 + LGC + BiMamba",
        device,
        amp_enabled,
        is_bimamba=True,
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
    save_csv(
        OUTPUT.result_dir / "test_predictions.csv", branch_predictions, prediction_fields
    )
    length_rows = compute_length_group_metrics(
        branch_predictions, "ESM2 + LGC + BiMamba", threshold
    )
    summaries = [summary_row("ESM2 + LGC + BiMamba", branch_metrics, branch_checkpoint)]
    all_metrics: Dict[str, Any] = {"ESM2 + LGC + BiMamba": branch_metrics}

    # 可选公平对比：基线同样只使用其 checkpoint 中保存的验证集阈值。
    if INFERENCE.baseline_checkpoint is not None:
        # 释放分支模型显存，避免同时驻留两个模型影响基线显存测量。
        del branch_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        baseline_checkpoint = load_checkpoint(INFERENCE.baseline_checkpoint, device)
        baseline_model = build_baseline_model(baseline_checkpoint).to(device)
        baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
        baseline_threshold = float(baseline_checkpoint["best_threshold"])
        baseline_predictions, baseline_runtime, baseline_metrics = predict_test_set(
            baseline_model,
            test_loader,
            baseline_threshold,
            "ESM2 + LGC + Transformer",
            device,
            amp_enabled,
            is_bimamba=False,
        )
        save_csv(
            OUTPUT.result_dir / "baseline_test_predictions.csv",
            baseline_predictions,
            prediction_fields,
        )
        runtime_rows.extend(baseline_runtime)
        length_rows.extend(
            compute_length_group_metrics(
                baseline_predictions, "ESM2 + LGC + Transformer", baseline_threshold
            )
        )
        summaries.append(
            summary_row("ESM2 + LGC + Transformer", baseline_metrics, baseline_checkpoint)
        )
        all_metrics["ESM2 + LGC + Transformer"] = baseline_metrics
        delta_fields = ("mcc", "aupr", "auc", "f1")
        save_csv(
            OUTPUT.result_dir / "comparison_deltas.csv",
            [
                {
                    "comparison": "BiMamba - Transformer",
                    **{
                        f"delta_{name}": branch_metrics[name] - baseline_metrics[name]
                        for name in delta_fields
                    },
                }
            ],
            ["comparison", "delta_mcc", "delta_aupr", "delta_auc", "delta_f1"],
        )

    save_csv(
        OUTPUT.result_dir / "runtime_by_length.csv",
        runtime_rows,
        [
            "method",
            "sample_id",
            "protein_length",
            "inference_time_seconds",
            "peak_gpu_memory_mb",
        ],
    )
    save_csv(
        OUTPUT.result_dir / "length_group_metrics.csv",
        length_rows,
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
    print(f"测试完成，固定验证阈值={threshold:.2f}")
    print(f"MCC={branch_metrics['mcc']:.4f}, AUPR={branch_metrics['aupr']:.4f}, AUC={branch_metrics['auc']:.4f}")
    print(f"结果目录：{OUTPUT.result_dir}")


if __name__ == "__main__":
    main()
