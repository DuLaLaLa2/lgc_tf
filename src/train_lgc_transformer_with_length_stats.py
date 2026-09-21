import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from src.data.rna_dataset import RNABindingDataset, pad_collate_fn
from src.losses import BinaryFocalLossWithLogits
from src.models.lgc_transformer_model import ESM2LGCTransformer
from src.utils.metrics import compute_binary_metrics, find_best_threshold


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/root/autodl-tmp/features/merged")

TRAIN_DIR = DATA_ROOT / "RNA-495_Train"
# VAL_DIR = DATA_ROOT / "RNA_Valid"
VAL_DIR = ""
TEST_DIR = DATA_ROOT / "RNA_1346_New"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "checkpoints"

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


def estimate_positive_alpha(dataset):
    pos = 0.0
    total = 0.0
    for i in range(len(dataset)):
        y = dataset[i]["y"]
        pos += float(y.sum().item())
        total += float(y.numel())

    neg = total - pos
    pos_ratio = pos / max(total, 1.0)
    alpha = neg / max(total, 1.0)
    return alpha, pos_ratio, pos, neg


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

    logits = model(
        x=x,
        coords=coords,
        mask=mask,
        coord_mask=coord_mask,
    )
    return logits, y, mask


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
    ("1500-2000", lambda length: 1500 <= length <= 2000),
    (">2000", lambda length: length > 2000),
)


def get_length_bin(length: int) -> str:
    """Map one protein length to a non-overlapping length interval."""
    for bin_name, contains in LENGTH_BINS:
        if contains(length):
            return bin_name
    raise ValueError(f"Invalid protein length: {length}")


def evaluate_by_length(model, loader, threshold: float):
    """Calculate residue-level MCC after grouping proteins by sequence length.

    The validation-selected ``threshold`` is fixed for every length bin. Within
    a bin, residue labels and probabilities from all proteins are concatenated,
    matching the residue-level metric convention used by ``evaluate``.
    """
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
                    y[sample_index][sample_mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                grouped[bin_name]["probs"].append(
                    probabilities[sample_index][sample_mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
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
            metrics = compute_binary_metrics(
                y_true=y_true,
                y_prob=y_prob,
                threshold=threshold,
            )
            mcc = float(metrics["mcc"])
            residue_count = int(y_true.size)

        rows.append(
            {
                "length_bin": bin_name,
                "protein_count": int(group["protein_count"]),
                "residue_count": residue_count,
                "transformer_mcc": mcc,
            }
        )
    return rows


def print_length_statistics(rows):
    print("=" * 80)
    print("Transformer MCC by Protein Length")
    print("=" * 80)
    print(
        f"{'Length bin':<16}"
        f"{'Proteins':>12}"
        f"{'Residues':>14}"
        f"{'Transformer MCC':>20}"
    )
    print("-" * 80)
    for row in rows:
        mcc_text = (
            "NA"
            if np.isnan(row["transformer_mcc"])
            else f"{row['transformer_mcc']:.6f}"
        )
        print(
            f"{row['length_bin']:<16}"
            f"{row['protein_count']:>12d}"
            f"{row['residue_count']:>14d}"
            f"{mcc_text:>20}"
        )
    print("=" * 80)


def save_length_statistics(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "length_bin",
                "protein_count",
                "residue_count",
                "transformer_mcc",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train extracted ESM2 + LGC + Transformer.")
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
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=USE_AMP)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def main():
    global USE_AMP

    args = parse_args()
    USE_AMP = args.amp
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ESM2 + LGC + Transformer Training")
    print("=" * 80)
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"TRAIN_DIR  : {args.train_dir}")
    print(f"VAL_DIR    : {args.val_dir}")
    print(f"TEST_DIR   : {args.test_dir}")
    print("MODEL      : ESM2 + LGC + Transformer")
    print(f"DEVICE     : {DEVICE}")
    print("=" * 80)

    full_train_dataset = RNABindingDataset(args.train_dir, max_len=args.max_len)
    test_dataset = RNABindingDataset(args.test_dir, max_len=args.max_len)

    if has_feature_files(args.val_dir):
        train_dataset = full_train_dataset
        val_dataset = RNABindingDataset(args.val_dir, max_len=args.max_len)
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

    alpha, pos_ratio, pos_count, neg_count = estimate_positive_alpha(full_train_dataset)
    print("-" * 80)
    print(f"Positive residues : {int(pos_count)}")
    print(f"Negative residues : {int(neg_count)}")
    print(f"Positive ratio    : {pos_ratio:.6f}")
    print(f"Focal alpha(pos)  : {alpha:.6f}")
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
    criterion = BinaryFocalLossWithLogits(alpha=alpha, gamma=2.0, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=amp_enabled())

    best_val_mcc = -1e9
    best_epoch = -1
    best_threshold = 0.5
    bad_epochs = 0
    ckpt_path = args.output_dir / "lgc_transformer_best-495-1346.pt"

    print("【开始训练...】")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            with autocast(device_type=DEVICE.type, enabled=amp_enabled()):
                logits, y, mask = forward_batch(model, batch)
                loss = criterion(logits[mask], y[mask]) / args.accum_steps

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
                        "model_type": "lgc_transformer",
                        "input_dim": INPUT_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS,
                        "ffn_dim": FFN_DIM,
                        "dropout": DROPOUT,
                        "batch_size": args.batch_size,
                        "accum_steps": args.accum_steps,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "focal_alpha": alpha,
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

    if args.max_len is not None:
        print(
            "WARNING: --max-len is set, so length bins use truncated lengths. "
            "Use --max-len None (the default) for true protein lengths."
        )

    length_rows = evaluate_by_length(
        model=model,
        loader=test_loader,
        threshold=best_threshold,
    )
    print_length_statistics(length_rows)

    length_stats_path = args.output_dir / "transformer_length_mcc-495-1346.csv"
    save_length_statistics(length_rows, length_stats_path)
    print(f"Length-stratified results saved to: {length_stats_path}")


if __name__ == "__main__":
    main()
