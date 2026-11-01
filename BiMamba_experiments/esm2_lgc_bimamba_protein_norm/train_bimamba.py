"""训练入口（按蛋白归一化 + 全局加权 Focal Loss）。

本目录是 esm2_lgc_bimamba 的分支实验，唯一目标是「在与 Transformer 基线
完全一致的损失设置下，单独比较 BiMamba 与 Transformer 的全局序列建模能力」。
因此除损失外，数据、LGC、随机种子、训练轮数、batch、梯度累积和 early stopping
均与基线保持不变。

相对原 esm2_lgc_bimamba 的改动只有损失：
    1. 损失统一换成 src.loss.GlobalWeightedFocalLossWithLogits
       （全局类别权重 pos_weight=sqrt(N_neg/N_pos) + focal 难样本权重）；
    2. 聚合方式改为「按蛋白归一化」：每条蛋白先取自身残基损失平均，再对
       batch 内蛋白等权平均，避免长序列稀疏正类被自身大量负类稀释。
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
from model_bimamba import ESM2LGCBiMamba

from src.loss import GlobalWeightedFocalLossWithLogits, compute_sqrt_pos_weight

from utils_bimamba import (
    compute_binary_metrics,
    count_parameters,
    find_best_mcc_threshold,
    save_csv,
    save_json,
    set_reproducible_seed,
    split_indices,
)


def configure_logger(log_path: Path) -> logging.Logger:
    """创建同时写入控制台和 UTF-8 文件的训练日志器。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bimamba_protein_norm_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def compute_class_weight(dataset: FullLengthRNADataset) -> Tuple[float, int, int]:
    """按训练集正负残基数计算全局正类权重 ``pos_weight``，输出 ``(pos_weight, 正数, 负数)``。"""

    positive_count = 0
    residue_count = 0
    for sample_index in range(len(dataset)):
        labels = dataset[sample_index]["y"]
        positive_count += int(labels.sum().item())
        residue_count += labels.numel()
    negative_count = residue_count - positive_count
    pos_weight = compute_sqrt_pos_weight(
        positive_count=float(positive_count),
        negative_count=float(negative_count),
        max_pos_weight=5.0,
    )
    return pos_weight, positive_count, negative_count


def per_protein_losses(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """逐蛋白计算损失，返回 [B] 形状的张量（每条蛋白自身残基平均）。"""

    losses = []
    for b in range(logits.shape[0]):
        m = mask[b].bool()
        if m.sum() == 0:
            continue
        losses.append(criterion(logits[b][m], y[b][m]))
    return torch.stack(losses)


def create_datasets() -> Tuple[Any, Any, FullLengthRNADataset, Dict[str, Any]]:
    """读取训练特征并确定验证划分。

    输出：训练集、验证集、完整训练集和可保存的划分清单。
    """

    full_dataset = FullLengthRNADataset(DATA.train_dir)
    if has_feature_files(DATA.val_dir):
        train_dataset = full_dataset
        validation_dataset = FullLengthRNADataset(DATA.val_dir)
        manifest = {
            "protocol": "explicit_validation_directory",
            "seed": TRAIN.seed,
            "train_files": [str(path) for path in full_dataset.files],
            "validation_files": [str(path) for path in validation_dataset.files],
        }
    else:
        train_indices, validation_indices = split_indices(
            len(full_dataset), DATA.val_ratio, TRAIN.seed
        )
        train_dataset = Subset(full_dataset, train_indices)
        validation_dataset = Subset(full_dataset, validation_indices)
        manifest = {
            "protocol": "seeded_train_validation_split",
            "seed": TRAIN.seed,
            "validation_ratio": DATA.val_ratio,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "train_files": [str(full_dataset.files[index]) for index in train_indices],
            "validation_files": [str(full_dataset.files[index]) for index in validation_indices],
        }
    return train_dataset, validation_dataset, full_dataset, manifest


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """把模型所需张量移动到设备，输出同名张量字典。"""

    keys = ("x", "coords", "coord_mask", "y", "mask", "lengths")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def forward_model(
    model: ESM2LGCBiMamba,
    tensors: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """执行一次模型前向，输入 batch 张量字典，输出 logits[B,L]。"""

    return model(
        x=tensors["x"],
        coords=tensors["coords"],
        mask=tensors["mask"],
        coord_mask=tensors["coord_mask"],
        lengths=tensors["lengths"],
    )


def oom_details(batch: Dict[str, Any], device: torch.device) -> str:
    """生成 OOM 诊断文本，包含样本 ID、长度和当前显存。"""

    memory_mb = (
        torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    return (
        f"sample_ids={batch['ids']}, lengths={batch['lengths'].tolist()}, "
        f"allocated_gpu_memory={memory_mb:.1f}MB"
    )


def train_one_epoch(
    model: ESM2LGCBiMamba,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """训练一个 epoch，输出按蛋白归一化的平均损失。"""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for step, batch in enumerate(loader, start=1):
        tensors = move_batch(batch, device)
        try:
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = forward_model(model, tensors)
                raw_loss = per_protein_losses(
                    logits, tensors["y"], tensors["mask"], criterion
                ).mean()
                loss = raw_loss / TRAIN.accumulation_steps
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(f"训练 batch 发生 CUDA OOM：{oom_details(batch, device)}") from error

        if step % TRAIN.accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(raw_loss.detach().item())
    return loss_sum / max(len(loader), 1)


def predict_validation(
    model: ESM2LGCBiMamba,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    fixed_threshold: float | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """预测验证集并计算指标。

    输入：模型、验证 loader、损失和可选固定阈值。
    输出：指标字典与逐残基预测行。阈值为空时仅在验证集最大化 MCC。
    """

    model.eval()
    labels: List[float] = []
    probabilities: List[float] = []
    prediction_rows: List[Dict[str, Any]] = []
    weighted_loss = 0.0
    valid_count = 0

    with torch.no_grad():
        for batch in loader:
            tensors = move_batch(batch, device)
            try:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = forward_model(model, tensors)
                    protein_losses = per_protein_losses(
                        logits, tensors["y"], tensors["mask"], criterion
                    )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"验证 batch 发生 CUDA OOM：{oom_details(batch, device)}"
                ) from error
            weighted_loss += float(protein_losses.sum().item())
            valid_count += int(protein_losses.numel())
            batch_probabilities = torch.sigmoid(logits).detach().cpu()
            batch_labels = batch["y"]

            for batch_index, length_value in enumerate(batch["lengths"].tolist()):
                length = int(length_value)
                for residue_index in range(length):
                    label = float(batch_labels[batch_index, residue_index].item())
                    probability = float(batch_probabilities[batch_index, residue_index].item())
                    labels.append(label)
                    probabilities.append(probability)
                    prediction_rows.append(
                        {
                            "sample_id": batch["ids"][batch_index],
                            "file": batch["files"][batch_index],
                            "protein_length": length,
                            "residue_index": residue_index + 1,
                            "label": int(label),
                            "probability": probability,
                        }
                    )

    if fixed_threshold is None:
        threshold, metrics = find_best_mcc_threshold(
            labels,
            probabilities,
            TRAIN.threshold_start,
            TRAIN.threshold_end,
            TRAIN.threshold_step,
        )
    else:
        threshold = fixed_threshold
        metrics = compute_binary_metrics(labels, probabilities, threshold)
    metrics["loss"] = weighted_loss / max(valid_count, 1)
    for row in prediction_rows:
        row["prediction"] = int(row["probability"] >= threshold)
        row["threshold"] = threshold
    return metrics, prediction_rows


def main() -> None:
    """执行完整训练和验证；不会创建或读取测试集 DataLoader。"""

    for output_dir in (OUTPUT.checkpoint_dir, OUTPUT.log_dir, OUTPUT.result_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(OUTPUT.log_dir / "train_bimamba.log")
    set_reproducible_seed(TRAIN.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = TRAIN.use_amp and device.type == "cuda"

    train_dataset, validation_dataset, full_dataset, split_manifest = create_datasets()
    save_json(OUTPUT.result_dir / "data_split.json", split_manifest)
    pos_weight, positive_count, negative_count = compute_class_weight(full_dataset)

    loader_kwargs = {
        "batch_size": TRAIN.batch_size,
        "num_workers": TRAIN.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": pad_collate_full_length,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = ESM2LGCBiMamba(**asdict(MODEL)).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    criterion = GlobalWeightedFocalLossWithLogits(
        pos_weight=pos_weight,
        gamma=TRAIN.focal_gamma,
        reduction="mean",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = OUTPUT.checkpoint_dir / OUTPUT.best_checkpoint_name

    logger.info("设备=%s, 训练/验证样本=%d/%d", device, len(train_dataset), len(validation_dataset))
    logger.info("正/负残基=%d/%d, pos_weight=%.6f", positive_count, negative_count, pos_weight)
    logger.info("总参数=%d, 可训练参数=%d", total_parameters, trainable_parameters)

    best_mcc = -float("inf")
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, TRAIN.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, amp_enabled
            )
            validation_metrics, _ = predict_validation(
                model, validation_loader, criterion, device, amp_enabled
            )
        except RuntimeError:
            logger.exception("训练或验证中止；如果是 OOM，日志已记录样本 ID、长度和显存。")
            raise

        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )
        current_threshold = float(validation_metrics["threshold"])
        improved = validation_metrics["mcc"] > best_mcc
        if improved:
            best_mcc = float(validation_metrics["mcc"])
            best_threshold = current_threshold
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mcc": best_mcc,
                    "best_threshold": best_threshold,
                    "model_config": asdict(MODEL),
                    "training_config": asdict(TRAIN),
                    "loss_type": "global_weighted_focal_protein_norm",
                    "pos_weight": pos_weight,
                    "focal_gamma": TRAIN.focal_gamma,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "split_manifest": split_manifest,
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_metrics["loss"],
                "val_mcc": validation_metrics["mcc"],
                "val_aupr": validation_metrics["aupr"],
                "val_auc": validation_metrics["auc"],
                "threshold": current_threshold,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "peak_gpu_memory_mb": peak_memory_mb,
                "epoch_seconds": epoch_seconds,
                "is_best": improved,
            }
        )
        save_csv(OUTPUT.log_dir / "train_history.csv", history, list(history[0].keys()))
        logger.info(
            "Epoch %03d | train=%.6f val=%.6f MCC=%.4f AUPR=%.4f AUC=%.4f "
            "thr=%.2f lr=%.2e GPU=%.1fMB time=%.1fs%s",
            epoch,
            train_loss,
            validation_metrics["loss"],
            validation_metrics["mcc"],
            validation_metrics["aupr"],
            validation_metrics["auc"],
            current_threshold,
            optimizer.param_groups[0]["lr"],
            peak_memory_mb,
            epoch_seconds,
            " BEST" if improved else "",
        )
        if epochs_without_improvement >= TRAIN.patience:
            logger.info("Early stopping：最佳 epoch=%d", best_epoch)
            break

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, prediction_rows = predict_validation(
        model, validation_loader, criterion, device, amp_enabled, fixed_threshold=best_threshold
    )
    validation_metrics.update(
        best_epoch=best_epoch,
        best_val_mcc=best_mcc,
        best_threshold=best_threshold,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )
    save_json(OUTPUT.result_dir / "validation_metrics.json", validation_metrics)
    save_csv(
        OUTPUT.result_dir / "validation_predictions.csv",
        prediction_rows,
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
            "model": "Sliding-window ESM2-3B + LGC + BiMamba (protein-norm focal)",
            "checkpoint": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_mcc": best_mcc,
            "best_threshold": best_threshold,
            "loss_type": "global_weighted_focal_protein_norm",
            "pos_weight": pos_weight,
            "model_config": asdict(MODEL),
            "training_config": asdict(TRAIN),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
    )
    logger.info("训练完成：checkpoint=%s", checkpoint_path)
    logger.info("最佳 epoch=%d, Val MCC=%.4f, threshold=%.2f", best_epoch, best_mcc, best_threshold)


if __name__ == "__main__":
    main()