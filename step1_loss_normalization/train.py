"""
Step 1 实验：损失层改造 —— 少数类关注 + 长序列。

在「不修改 src 现有代码」的前提下，复用：
    - src.models.lgc_transformer_model.ESM2LGCTransformer（模型骨架）
    - src.loss.GlobalWeightedFocalLossWithLogits（全局类别权重 + focal 难样本权重）
    - src.loss.compute_sqrt_pos_weight（按训练集正负比例算温和权重）
    - src.data.rna_dataset / src.utils.metrics（数据与评测）

相对原基线的唯一、可控改动是损失聚合方式（--loss-norm）：

    residue : 对所有有效残基取平均（等价于原 GlobalWeightedFocalLoss 的 mean）
               —— 作为「对照」基线，复现原行为。

    protein : 每条蛋白先取自身残基损失的平均，再对 batch 内蛋白取平均。
               —— 让长短蛋白等权，避免长序列稀疏正类被自身大量负类稀释，
                  这是 Step 1 的目标改动。

同时修正原 train_stage2_baseline.py 中 estimate_class_statistics 返回值
与解包顺序不一致、导致 pos_weight 计算错误的问题。
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rna_dataset import RNABindingDataset, pad_collate_fn
from src.loss import GlobalWeightedFocalLossWithLogits, compute_sqrt_pos_weight
from src.models.lgc_transformer_model import ESM2LGCTransformer
from src.utils.metrics import compute_binary_metrics, find_best_threshold


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/root/autodl-tmp/features/merged")

TRAIN_DIR = DATA_ROOT / "RNA-495_Train"
VAL_DIR = ""
TEST_DIR = DATA_ROOT / "RNA-117_Test"
OUTPUT_DIR = BASE_DIR / "outputs"

SEED = 42
VAL_RATIO = 0.2

INPUT_DIM = 2560
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1

EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 2
ACCUM_STEPS = 4

LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
FOCAL_GAMMA = 2.0

USE_AMP = True
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled() -> bool:
    return USE_AMP and DEVICE.type == "cuda"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val(n_samples: int, val_ratio: float, seed: int):
    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(n_samples * val_ratio))
    return indices[val_size:], indices[:val_size]


def has_feature_files(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.glob("*.pt"))


def estimate_class_statistics(dataset):
    """统计正/负类残基数，返回 (pos_count, neg_count, pos_ratio, imbalance_ratio)。"""
    pos_count = 0.0
    total_count = 0.0
    for i in range(len(dataset)):
        y = dataset[i]["y"]
        pos_count += float(y.sum().item())
        total_count += float(y.numel())

    neg_count = total_count - pos_count
    pos_ratio = pos_count / max(total_count, 1.0)
    imbalance_ratio = neg_count / max(pos_count, 1.0)
    return pos_count, neg_count, pos_ratio, imbalance_ratio


def build_model():
    return ESM2LGCTransformer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        sde_dim=16,
        cutoff=30.0,
        k_neighbors=16,
    )


def forward_batch(model, batch):
    x = batch["x"].to(DEVICE, non_blocking=True)
    y = batch["y"].to(DEVICE, non_blocking=True)
    mask = batch["mask"].to(DEVICE, non_blocking=True)
    coords = batch["coords"].to(DEVICE, non_blocking=True)
    coord_mask = batch["coord_mask"].to(DEVICE, non_blocking=True)

    logits = model(x=x, coords=coords, mask=mask, coord_mask=coord_mask)
    return logits, y, mask


def compute_loss(logits, y, mask, criterion, loss_norm: str) -> torch.Tensor:
    """按指定归一化方式聚合逐残基损失。

    residue : 把所有有效残基拍平后取平均（等价于 criterion 的 mean）。
    protein : 每条蛋白先取自身残基平均，再对 batch 内蛋白取平均。
    """
    if loss_norm == "protein":
        losses = []
        for b in range(logits.shape[0]):
            m = mask[b].bool()
            if m.sum() == 0:
                continue
            losses.append(criterion(logits[b][m], y[b][m]))
        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()

    return criterion(logits[mask], y[mask])


def collect_predictions(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=DEVICE.type, enabled=amp_enabled()):
                logits, y, mask = forward_batch(model, batch)

            all_probs.append(torch.sigmoid(logits[mask]).detach().cpu().numpy())
            all_labels.append(y[mask].detach().cpu().numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def evaluate(model, loader, threshold=None):
    y_true, y_prob = collect_predictions(model, loader)
    if threshold is None:
        threshold, metrics = find_best_threshold(y_true=y_true, y_prob=y_prob, metric_name="mcc")
    else:
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    return metrics, threshold


LENGTH_BINS = (
    ("<500", lambda length: length < 500),
    ("500-1000", lambda length: 500 <= length < 1000),
    ("1000-1500", lambda length: 1000 <= length < 1500),
    # ("1500-2000", lambda length: 1500 <= length <= 2000),
    (">1500", lambda length: length >= 1500),
)


def get_length_bin(length: int) -> str:
    for bin_name, contains in LENGTH_BINS:
        if contains(length):
            return bin_name
    raise ValueError(f"Invalid protein length: {length}")


def evaluate_by_length(model, loader, threshold: float):
    model.eval()
    grouped = {
        bin_name: {"protein_count": 0, "labels": [], "probs": []}
        for bin_name, _ in LENGTH_BINS
    }

    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=DEVICE.type, enabled=amp_enabled()):
                logits, y, mask = forward_batch(model, batch)

            probabilities = torch.sigmoid(logits)
            for sample_index in range(mask.shape[0]):
                sample_mask = mask[sample_index].bool()
                protein_length = int(sample_mask.sum().item())
                bin_name = get_length_bin(protein_length)

                grouped[bin_name]["protein_count"] += 1
                grouped[bin_name]["labels"].append(
                    y[sample_index][sample_mask].detach().cpu().numpy().reshape(-1)
                )
                grouped[bin_name]["probs"].append(
                    probabilities[sample_index][sample_mask].detach().cpu().numpy().reshape(-1)
                )

    rows = []
    for bin_name, _ in LENGTH_BINS:
        group = grouped[bin_name]
        if group["protein_count"] == 0:
            mcc = float("nan")
            residue_count = 0
        else:
            y_true = np.concatenate(group["labels"], axis=0)
            y_prob = np.concatenate(group["probs"], axis=0)
            metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
            mcc = float(metrics["mcc"])
            residue_count = int(y_true.size)

        rows.append(
            {
                "length_bin": bin_name,
                "protein_count": int(group["protein_count"]),
                "residue_count": residue_count,
                "mcc": mcc,
            }
        )
    return rows


def save_length_statistics(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("length_bin", "protein_count", "residue_count", "mcc"),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_length_statistics(rows):
    print("=" * 80)
    print("MCC by Protein Length")
    print("=" * 80)
    for row in rows:
        mcc_text = "NA" if np.isnan(row["mcc"]) else f"{row['mcc']:.6f}"
        print(
            f"{row['length_bin']:<16}"
            f"{row['protein_count']:>12d}"
            f"{row['residue_count']:>14d}"
            f"{mcc_text:>20}"
        )
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Step1: global weighted focal + per-protein normalization.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--loss-norm", type=str, choices=["residue", "protein"], default="protein")
    parser.add_argument("--amp", dest="amp", action="store_true", default=USE_AMP)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def main():
    global USE_AMP

    args = parse_args()
    USE_AMP = args.amp
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "length_stats").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step1 experiment: ESM2 + LGC + Transformer (loss normalization)")
    print("=" * 80)
    print(f"loss-norm    : {args.loss_norm}")
    print(f"train-dir    : {args.train_dir}")
    print(f"val-dir      : {args.val_dir}")
    print(f"test-dir     : {args.test_dir}")
    print(f"device       : {DEVICE}")
    print("=" * 80)

    full_train_dataset = RNABindingDataset(args.train_dir, max_len=None)
    test_dataset = RNABindingDataset(args.test_dir, max_len=None)

    if has_feature_files(args.val_dir):
        train_dataset = full_train_dataset
        val_dataset = RNABindingDataset(args.val_dir, max_len=None)
        split_protocol = f"explicit validation directory: {args.val_dir}"
    else:
        train_indices, val_indices = split_train_val(
            n_samples=len(full_train_dataset),
            val_ratio=VAL_RATIO,
            seed=SEED,
        )
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        split_protocol = f"train split fallback: train/val = 8/2, seed={SEED}"

    print(f"Full train samples: {len(full_train_dataset)}")
    print(f"Train samples     : {len(train_dataset)}")
    print(f"Val samples       : {len(val_dataset)}")
    print(f"Test samples      : {len(test_dataset)}")
    print(f"Split protocol    : {split_protocol}")

    pos_count, neg_count, pos_ratio, imbalance_ratio = estimate_class_statistics(train_dataset)
    pos_weight = compute_sqrt_pos_weight(
        positive_count=pos_count,
        negative_count=neg_count,
        max_pos_weight=5.0,
    )

    print("-" * 80)
    print(f"Positive residues : {int(pos_count)}")
    print(f"Negative residues : {int(neg_count)}")
    print(f"Positive ratio    : {pos_ratio:.6f}")
    print(f"Neg / Pos ratio   : {imbalance_ratio:.4f}")
    print(f"Global pos weight : {pos_weight:.4f}")
    print(f"Focal gamma       : {FOCAL_GAMMA:.2f}")
    print("-" * 80)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "collate_fn": pad_collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model().to(DEVICE)
    criterion = GlobalWeightedFocalLossWithLogits(
        pos_weight=pos_weight,
        gamma=FOCAL_GAMMA,
        reduction="mean",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=amp_enabled())

    best_val_mcc = -1e9
    best_epoch = -1
    best_threshold = 0.5
    bad_epochs = 0
    ckpt_path = args.output_dir / "checkpoints" / f"lgc_transformer_step1_{args.loss_norm}_2.pt"

    print("Start training...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            with autocast(device_type=DEVICE.type, enabled=amp_enabled()):
                logits, y, mask = forward_batch(model, batch)
                loss = compute_loss(logits, y, mask, criterion, args.loss_norm) / args.accum_steps

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.accum_steps
            total_steps += 1

        avg_train_loss = total_loss / max(total_steps, 1)
        val_metrics, val_threshold = evaluate(model=model, loader=val_loader, threshold=None)
        improved = val_metrics["mcc"] > best_val_mcc

        if improved:
            best_val_mcc = val_metrics["mcc"]
            best_epoch = epoch
            best_threshold = val_threshold
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mcc": best_val_mcc,
                    "best_threshold": best_threshold,
                    "config": {
                        "experiment": "step1_loss_normalization",
                        "loss_norm": args.loss_norm,
                        "loss_type": "global_weighted_focal",
                        "loss_pos_weight": pos_weight,
                        "loss_gamma": FOCAL_GAMMA,
                        "train_positive_count": int(pos_count),
                        "train_negative_count": int(neg_count),
                        "train_positive_ratio": pos_ratio,
                        "train_neg_pos_ratio": imbalance_ratio,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS,
                        "ffn_dim": FFN_DIM,
                        "split_protocol": split_protocol,
                    },
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_train_loss:.6f} | "
            f"val_mcc={val_metrics['mcc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_aupr={val_metrics['aupr']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"thr={val_threshold:.2f} | "
            f"{'BEST' if improved else ''}"
        )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print("=" * 80)
    print("Training finished.")
    print(f"Best val MCC      : {best_val_mcc:.4f}")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best threshold    : {best_threshold:.2f}")
    print(f"Best checkpoint   : {ckpt_path}")
    print("=" * 80)

    print("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, _ = evaluate(model=model, loader=test_loader, threshold=best_threshold)

    print("=" * 80)
    print("Independent Test Results")
    print("=" * 80)
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key:12s}: {value:.6f}")
        else:
            print(f"{key:12s}: {value}")
    print("=" * 80)

    length_rows = evaluate_by_length(model=model, loader=test_loader, threshold=best_threshold)
    print_length_statistics(length_rows)

    length_stats_path = (
        args.output_dir / "length_stats" / f"length_mcc_step1_{args.loss_norm}.csv"
    )
    save_length_statistics(length_rows, length_stats_path)
    print(f"Length-stratified results saved to: {length_stats_path}")


if __name__ == "__main__":
    main()