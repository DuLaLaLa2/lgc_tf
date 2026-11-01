import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from data.rna_dataset import RNABindingDataset, pad_collate_fn
from models.lgc_transformer_model import ESM2LGCTransformer
from utils.metrics import compute_binary_metrics


PROJECT_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path("/root/autodl-tmp/features/merged")

TEST_DIR = DATA_ROOT / "RNA_1346_New"
# TEST_DIR = DATA_ROOT / "RNA-117_Test"
TEST_DIR = "/root/autodl-tmp/features/feat_extra"
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "lgc_transformer_best-495-1346_loss1.pt"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions"

INPUT_DIM = 2560
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1

BATCH_SIZE = 2
USE_AMP = True
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Length 1500 belongs to 1000-1500; the final bin is strictly >1500.
LENGTH_BINS = (
    ("<500", lambda length: length < 500),
    ("500-1000", lambda length: 500 <= length < 1000),
    ("1000-1500", lambda length: 1000 <= length <= 1500),
    (">1500", lambda length: length > 1500),
)


def amp_enabled() -> bool:
    return USE_AMP and DEVICE.type == "cuda"


def build_model(config=None):
    """Build the model, preferring architecture values stored in checkpoint."""
    config = config or {}
    return ESM2LGCTransformer(
        input_dim=int(config.get("input_dim", INPUT_DIM)),
        hidden_dim=int(config.get("hidden_dim", HIDDEN_DIM)),
        num_layers=int(config.get("num_layers", NUM_LAYERS)),
        num_heads=int(config.get("num_heads", NUM_HEADS)),
        ffn_dim=int(config.get("ffn_dim", FFN_DIM)),
        dropout=float(config.get("dropout", DROPOUT)),
        sde_dim=int(config.get("sde_dim", 16)),
        cutoff=float(config.get("cutoff", 30.0)),
        k_neighbors=int(config.get("k_neighbors", 16)),
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

    if not all_labels:
        raise ValueError("The test DataLoader produced no samples")
    return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def evaluate(model, loader, threshold: float):
    y_true, y_prob = collect_predictions(model, loader)
    return compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )


def get_length_bin(length: int) -> str:
    for bin_name, contains in LENGTH_BINS:
        if contains(length):
            return bin_name
    raise ValueError(f"Invalid protein length: {length}")


def evaluate_by_length(model, loader, threshold: float):
    """Pool residue predictions within each protein-length bin and compute MCC."""
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


def print_metrics(metrics):
    print("=" * 80)
    print("Independent Test Results")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:12s}: {value:.6f}")
        else:
            print(f"{key:12s}: {value}")
    print("=" * 80)


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(
        description="Predict with a trained ESM2 + LGC + Transformer checkpoint."
    )
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=USE_AMP)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def main():
    global USE_AMP

    args = parse_args()
    USE_AMP = args.amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_len is not None:
        raise ValueError(
            "Length-stratified evaluation requires true protein lengths; "
            "do not set --max-len."
        )

    print("=" * 80)
    print("ESM2 + LGC + Transformer Prediction")
    print("=" * 80)
    print(f"TEST_DIR   : {args.test_dir}")
    print(f"CHECKPOINT : {args.checkpoint}")
    print(f"OUTPUT_DIR : {args.output_dir}")
    print(f"DEVICE     : {DEVICE}")
    print("=" * 80)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(config=config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.threshold is None:
        if "best_threshold" not in checkpoint:
            raise KeyError(
                "Checkpoint has no best_threshold; pass the validation-selected "
                "value explicitly with --threshold."
            )
        threshold = float(checkpoint["best_threshold"])
    else:
        threshold = float(args.threshold)

    test_dataset = RNABindingDataset(args.test_dir, max_len=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=pad_collate_fn,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Threshold   : {threshold:.6f}")

    test_metrics = evaluate(model=model, loader=test_loader, threshold=threshold)
    print_metrics(test_metrics)

    length_rows = evaluate_by_length(
        model=model,
        loader=test_loader,
        threshold=threshold,
    )
    print_length_statistics(length_rows)

    length_stats_path = args.output_dir / "transformer_length_mcc.csv"
    save_length_statistics(length_rows, length_stats_path)
    print(f"Length-stratified results saved to: {length_stats_path}")


if __name__ == "__main__":
    main()
