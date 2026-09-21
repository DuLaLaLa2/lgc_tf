"""并行分支训练入口：只训练、验证和保存产物，不读取测试集。

修改 config.py 后在 IDE 中直接运行，无命令行参数。
"""

import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from config import DATA, MODEL, OUTPUT, TRAIN
from dataset_adapter import FullLengthRNADataset, has_feature_files, pad_collate_full_length
from model_parallel import ESM2ParallelLGCBiMamba
from src.loss import GlobalWeightedFocalLossWithLogits,compute_sqrt_pos_weight
from utils_parallel import (
    best_mcc_threshold,
    binary_metrics,
    parameter_counts,
    save_csv,
    save_json,
    set_seed,
    split_indices,
)


def make_logger(path: Path) -> logging.Logger:
    """返回同时写控制台和 UTF-8 文件的日志器。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("parallel_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def estimate_class_statistics(dataset):
    pos_count = 0.0
    total_count = 0.0

    for i in range(len(dataset)):
        sample = dataset[i]

        y = sample["y"]

        pos_count += float(
            y.sum().item()
        )

        total_count += float(
            y.numel()
        )

    neg_count = (
        total_count - pos_count
    )

    pos_ratio = (
        pos_count
        / max(total_count, 1.0)
    )

    imbalance_ratio = (
        neg_count
        / max(pos_count, 1.0)
    )

    return imbalance_ratio, pos_ratio , pos_count, neg_count

def create_datasets() -> Tuple[Any, Any, FullLengthRNADataset, Dict[str, Any]]:
    """读取数据并复现验证划分，返回训练集、验证集、全集和划分清单。"""

    full_dataset = FullLengthRNADataset(DATA.train_dir)
    if has_feature_files(DATA.val_dir):
        train_dataset = full_dataset
        val_dataset = FullLengthRNADataset(DATA.val_dir)
        manifest = {
            "protocol": "explicit_validation_directory",
            "seed": TRAIN.seed,
            "train_files": [str(path) for path in full_dataset.files],
            "validation_files": [str(path) for path in val_dataset.files],
        }
    else:
        train_indices, val_indices = split_indices(len(full_dataset), DATA.val_ratio, TRAIN.seed)
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        manifest = {
            "protocol": "seeded_train_validation_split",
            "seed": TRAIN.seed,
            "validation_ratio": DATA.val_ratio,
            "train_indices": train_indices,
            "validation_indices": val_indices,
            "train_files": [str(full_dataset.files[index]) for index in train_indices],
            "validation_files": [str(full_dataset.files[index]) for index in val_indices],
        }
    return train_dataset, val_dataset, full_dataset, manifest


def focal_alpha(dataset: FullLengthRNADataset) -> Tuple[float, int, int]:
    """统计全集标签，返回正类 Focal 权重、正残基数和负残基数。"""

    positive = 0
    total = 0
    for index in range(len(dataset)):
        labels = dataset[index]["y"]
        positive += int(labels.sum().item())
        total += labels.numel()
    negative = total - positive
    return negative / max(total, 1), positive, negative


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """把前向所需张量移动至设备。"""

    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("x", "coords", "coord_mask", "y", "mask", "lengths")
    }


def forward(model: ESM2ParallelLGCBiMamba, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """输入 batch 张量字典，输出残基 logits[B,L]。"""

    return model(
        x=tensors["x"],
        coords=tensors["coords"],
        mask=tensors["mask"],
        coord_mask=tensors["coord_mask"],
        lengths=tensors["lengths"],
    )


def oom_message(batch: Dict[str, Any], device: torch.device) -> str:
    """输出包含样本、长度和当前显存的 OOM 上下文。"""

    memory = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return f"ids={batch['ids']}, lengths={batch['lengths'].tolist()}, GPU={memory:.1f}MB"


def train_epoch(
    model: ESM2ParallelLGCBiMamba,
    loader: DataLoader,
    criterion: GlobalWeightedFocalLossWithLogits,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """训练一个 epoch，返回平均 batch loss。"""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for step, batch in enumerate(loader, start=1):
        tensors = to_device(batch, device)
        try:
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = forward(model, tensors)
                raw_loss = criterion(logits[tensors["mask"]], tensors["y"][tensors["mask"]])
                scaled_loss = raw_loss / TRAIN.accumulation_steps
            scaler.scale(scaled_loss).backward()
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(f"训练 OOM：{oom_message(batch, device)}") from error

        if step % TRAIN.accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(raw_loss.detach().item())
    return loss_sum / max(len(loader), 1)


def evaluate_validation(
    model: ESM2ParallelLGCBiMamba,
    loader: DataLoader,
    criterion: GlobalWeightedFocalLossWithLogits,
    device: torch.device,
    amp_enabled: bool,
    fixed_threshold: float | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """预测验证集并返回指标和逐残基记录；阈值为空时在验证集搜索。"""

    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    rows: List[Dict[str, Any]] = []
    loss_sum = 0.0
    residue_count = 0
    with torch.no_grad():
        for batch in loader:
            tensors = to_device(batch, device)
            try:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = forward(model, tensors)
                    selected_logits = logits[tensors["mask"]]
                    selected_labels = tensors["y"][tensors["mask"]]
                    loss = criterion(selected_logits, selected_labels)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(f"验证 OOM：{oom_message(batch, device)}") from error

            loss_sum += float(loss.item()) * selected_labels.numel()
            residue_count += selected_labels.numel()
            batch_probabilities = torch.sigmoid(logits).float().cpu()
            for batch_index, length_value in enumerate(batch["lengths"].tolist()):
                length = int(length_value)
                for residue_index in range(length):
                    label = int(batch["y"][batch_index, residue_index].item())
                    probability = float(batch_probabilities[batch_index, residue_index].item())
                    labels.append(label)
                    probabilities.append(probability)
                    rows.append(
                        {
                            "sample_id": batch["ids"][batch_index],
                            "file": batch["files"][batch_index],
                            "protein_length": length,
                            "residue_index": residue_index + 1,
                            "label": label,
                            "probability": probability,
                        }
                    )

    if fixed_threshold is None:
        threshold, metrics = best_mcc_threshold(
            labels,
            probabilities,
            TRAIN.threshold_start,
            TRAIN.threshold_end,
            TRAIN.threshold_step,
        )
    else:
        threshold = fixed_threshold
        metrics = binary_metrics(labels, probabilities, threshold)
    metrics["loss"] = loss_sum / max(residue_count, 1)
    for row in rows:
        row["prediction"] = int(row["probability"] >= threshold)
        row["threshold"] = threshold
    return metrics, rows


def main() -> None:
    """完成训练和验证，并生成推理所需 checkpoint。"""

    for directory in (OUTPUT.checkpoint_dir, OUTPUT.log_dir, OUTPUT.result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logger = make_logger(OUTPUT.log_dir / "train_parallel.log")
    set_seed(TRAIN.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = TRAIN.use_amp and device.type == "cuda"

    train_dataset, val_dataset, full_dataset, manifest = create_datasets()
    save_json(OUTPUT.result_dir / "data_split.json", manifest)
    imbalance_ratio, pos_ratio, pos_count, neg_count = estimate_class_statistics(train_dataset)

    pos_weight = compute_sqrt_pos_weight(
        positive_count=pos_count,
        negative_count=neg_count,
        max_pos_weight=5.0,
    )
    print("-" * 80)
    print(f"Positive residues : " f"{int(pos_count)}")
    print(f"Negative residues : " f"{int(neg_count)}")
    print(f"Positive ratio    : "f"{pos_ratio:.6f}")
    print(f"Neg / Pos ratio   : "f"{imbalance_ratio:.4f}")
    print(f"Global pos weight : "f"{pos_weight:.4f}")
    print(f"Focal gamma       : "f"{TRAIN.focal_gamma:.2f}")
    print("-" * 80)


    alpha, positives, negatives = focal_alpha(full_dataset)
    loader_options = {
        "batch_size": TRAIN.batch_size,
        "num_workers": TRAIN.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": pad_collate_full_length,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = ESM2ParallelLGCBiMamba(**asdict(MODEL)).to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    criterion = GlobalWeightedFocalLossWithLogits(pos_weight=pos_weight, gamma=TRAIN.focal_gamma, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = OUTPUT.checkpoint_dir / OUTPUT.best_checkpoint_name
    logger.info("device=%s, train/val=%d/%d", device, len(train_dataset), len(val_dataset))
    logger.info("parameters total/trainable=%d/%d", total_parameters, trainable_parameters)

    best_mcc = -float("inf")
    best_threshold = 0.5
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, TRAIN.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, amp_enabled
            )
            val_metrics, _ = evaluate_validation(
                model, val_loader, criterion, device, amp_enabled
            )
        except RuntimeError:
            logger.exception("训练或验证中止；OOM 信息包含样本 ID、长度和显存。")
            raise

        elapsed = time.perf_counter() - epoch_start
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )
        improved = val_metrics["mcc"] > best_mcc
        if improved:
            best_mcc = float(val_metrics["mcc"])
            best_threshold = float(val_metrics["threshold"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mcc": best_mcc,
                    "best_threshold": best_threshold,
                    "model_config": asdict(MODEL),
                    "training_config": asdict(TRAIN),
                    "loss_type": "global_weighted_focal",
                    "loss_pos_weight": pos_weight,
                    "loss_gamma": TRAIN.focal_gamma,
                    "train_positive_count": int(pos_count),
                    "train_negative_count": int(neg_count),
                    "train_positive_ratio": pos_ratio,
                    "train_neg_pos_ratio": imbalance_ratio,
                    "total_parameters": total_parameters,
                    "split_manifest": manifest,
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "fusion": "concat_h_lgc_h_bimamba_without_projection",
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mcc": val_metrics["mcc"],
                "val_aupr": val_metrics["aupr"],
                "val_auc": val_metrics["auc"],
                "threshold": val_metrics["threshold"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "peak_gpu_memory_mb": peak_memory,
                "epoch_seconds": elapsed,
                "is_best": improved,
            }
        )
        save_csv(OUTPUT.log_dir / "train_history.csv", history, list(history[0].keys()))
        logger.info(
            "Epoch %03d | train=%.6f val=%.6f MCC=%.4f AUPR=%.4f AUC=%.4f "
            "thr=%.2f GPU=%.1fMB time=%.1fs%s",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["mcc"],
            val_metrics["aupr"],
            val_metrics["auc"],
            val_metrics["threshold"],
            peak_memory,
            elapsed,
            " BEST" if improved else "",
        )
        if bad_epochs >= TRAIN.patience:
            logger.info("Early stopping：best epoch=%d", best_epoch)
            break

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_rows = evaluate_validation(
        model, val_loader, criterion, device, amp_enabled, fixed_threshold=best_threshold
    )
    val_metrics.update(
        best_epoch=best_epoch,
        best_val_mcc=best_mcc,
        best_threshold=best_threshold,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )
    save_json(OUTPUT.result_dir / "validation_metrics.json", val_metrics)
    save_csv(
        OUTPUT.result_dir / "validation_predictions.csv",
        val_rows,
        [
            "sample_id",
            "file",
            "protein_length",
            "residue_index",
            "label",
            "probability",
            "prediction",
            "threshold",
        ],
    )
    save_json(
        OUTPUT.result_dir / "training_summary.json",
        {
            "model": "ESM2 + Parallel LGC/BiMamba + Concat",
            "checkpoint": str(checkpoint_path),
            "fusion": "H_final = Concat(H_lgc, H_bimamba), shape=[B,L,2d]",
            "best_epoch": best_epoch,
            "best_val_mcc": best_mcc,
            "best_threshold": best_threshold,
            "model_config": asdict(MODEL),
            "training_config": asdict(TRAIN),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
    )
    logger.info("完成：checkpoint=%s, best MCC=%.4f, threshold=%.2f", checkpoint_path, best_mcc, best_threshold)


if __name__ == "__main__":
    main()

