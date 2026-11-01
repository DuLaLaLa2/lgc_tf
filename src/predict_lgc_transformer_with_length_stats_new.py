import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from data.rna_dataset_new import RNABindingDataset, pad_collate_fn
from models.lgc_transformer_model import ESM2LGCTransformer
from utils.metrics import compute_binary_metrics


PROJECT_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path("/root/autodl-tmp/features/merged")

# 默认对额外提取的蛋白质特征执行预测（允许缺少 label）。
# TEST_DIR = Path("/root/autodl-tmp/features/feat_extra")
TEST_DIR = DATA_ROOT / "RNA_1346_New"
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "lgc_transformer_best-495-1346_loss1.pt"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions2"

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

# 保留原始长度分组规则：1500 归入 1000–1500 组。
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


def get_length_bin(length: int) -> str:
    for bin_name, contains in LENGTH_BINS:
        if contains(length):
            return bin_name
    raise ValueError(f"Invalid protein length: {length}")


def predict_and_evaluate(model, loader, threshold: float, output_dir: Path):
    """单次前向预测全部蛋白；只用有真实标签的蛋白计算分类指标和长度分组 MCC。"""
    model.eval()
    grouped = {
        bin_name: {"protein_count": 0, "labels": [], "probs": []}
        for bin_name, _ in LENGTH_BINS
    }
    all_labels = []
    all_probs = []
    labelled_proteins = 0
    unlabelled_proteins = 0

    # 无标签蛋白也会输出真实预测概率与阈值分类结果。
    residue_csv = output_dir / "residue_predictions.csv"
    protein_csv = output_dir / "protein_prediction_summary.csv"
    with residue_csv.open("w", newline="", encoding="utf-8-sig") as residue_handle, \
         protein_csv.open("w", newline="", encoding="utf-8-sig") as protein_handle:
        residue_writer = csv.writer(residue_handle)
        protein_writer = csv.writer(protein_handle)
        residue_writer.writerow((
            "protein_id", "source_file", "residue_position", "amino_acid",
            "binding_probability", "predicted_label", "true_label",
        ))
        protein_writer.writerow((
            "protein_id", "source_file", "protein_length", "length_bin",
            "has_true_label", "predicted_binding_residues", "predicted_binding_ratio",
        ))

        with torch.no_grad():
            for batch in loader:
                with autocast(device_type=DEVICE.type, enabled=amp_enabled()):
                    logits, y, mask = forward_batch(model, batch)

                probabilities = torch.sigmoid(logits)
                label_mask = batch["label_mask"].to(DEVICE, non_blocking=True)

                for sample_index in range(mask.shape[0]):
                    sample_mask = mask[sample_index].bool()
                    protein_length = int(sample_mask.sum().item())
                    bin_name = get_length_bin(protein_length)
                    protein_id = str(batch["ids"][sample_index])
                    source_file = batch["files"][sample_index]
                    sequence = batch["sequences"][sample_index]

                    # 转为 float32 后进行概率判定，避免自动混合精度的输出类型差异。
                    protein_probs = (probabilities[sample_index][sample_mask]
                                     .float().detach().cpu().numpy())
                    protein_preds = (protein_probs >= threshold).astype(np.int64)
                    has_true_label = bool(label_mask[sample_index][sample_mask].all().item())
                    true_labels = None

                    if has_true_label:
                        labelled_proteins += 1
                        true_labels = (y[sample_index][sample_mask]
                                       .detach().cpu().numpy())
                        all_labels.append(true_labels)
                        all_probs.append(protein_probs)
                        grouped[bin_name]["protein_count"] += 1
                        grouped[bin_name]["labels"].append(true_labels)
                        grouped[bin_name]["probs"].append(protein_probs)
                    else:
                        unlabelled_proteins += 1

                    protein_writer.writerow((
                        protein_id, source_file, protein_length, bin_name,
                        int(has_true_label), int(protein_preds.sum()),
                        f"{protein_preds.mean():.6f}",
                    ))
                    for position in range(protein_length):
                        amino_acid = sequence[position] if position < len(sequence) else ""
                        real_label = int(true_labels[position]) if has_true_label else ""
                        residue_writer.writerow((
                            protein_id, source_file, position + 1, amino_acid,
                            f"{protein_probs[position]:.8f}",
                            int(protein_preds[position]), real_label,
                        ))

    # 标签缺失时不计算 MCC、AUC 等指标，也不生成虚假的评估结果。
    if not all_labels:
        return None, None, labelled_proteins, unlabelled_proteins, residue_csv, protein_csv

    test_metrics = compute_binary_metrics(
        y_true=np.concatenate(all_labels, axis=0),
        y_prob=np.concatenate(all_probs, axis=0),
        threshold=threshold,
    )
    length_rows = []
    for bin_name, _ in LENGTH_BINS:
        group = grouped[bin_name]
        if group["protein_count"] == 0:
            mcc = float("nan")
            residue_count = 0
        else:
            labels = np.concatenate(group["labels"], axis=0)
            probs = np.concatenate(group["probs"], axis=0)
            metrics = compute_binary_metrics(
                y_true=labels, y_prob=probs, threshold=threshold,
            )
            mcc = float(metrics["mcc"])
            residue_count = int(labels.size)
        length_rows.append({
            "length_bin": bin_name,
            "protein_count": int(group["protein_count"]),
            "residue_count": residue_count,
            "transformer_mcc": mcc,
        })
    return test_metrics, length_rows, labelled_proteins, unlabelled_proteins, residue_csv, protein_csv


def print_metrics(metrics):
    print("=" * 80)
    print("Independent Test Results (proteins with true labels only)")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:12s}: {value:.6f}")
        else:
            print(f"{key:12s}: {value}")
    print("=" * 80)


def print_length_statistics(rows):
    print("=" * 80)
    print("Transformer MCC by Protein Length (labelled proteins only)")
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
            "NA" if np.isnan(row["transformer_mcc"])
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
            fieldnames=("length_bin", "protein_count", "residue_count", "transformer_mcc"),
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

    # 允许特征文件缺少 label：训练数据加载器仍默认要求真实标签。
    test_dataset = RNABindingDataset(args.test_dir, max_len=None, require_labels=False)
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

    (test_metrics, length_rows, labelled_count, unlabelled_count,
     residue_path, protein_path) = predict_and_evaluate(
        model=model, loader=test_loader, threshold=threshold, output_dir=args.output_dir
    )

    print(f"Proteins with labels   : {labelled_count}")
    print(f"Proteins without labels: {unlabelled_count}")
    print(f"Residue predictions    : {residue_path}")
    print(f"Protein summary        : {protein_path}")

    if test_metrics is None:
        print("No ground-truth labels: skipped MCC / F1 / AUC / AUPR and length-group evaluation.")
    else:
        print_metrics(test_metrics)
        print_length_statistics(length_rows)
        length_stats_path = args.output_dir / "transformer_length_mcc.csv"
        save_length_statistics(length_rows, length_stats_path)
        print(f"Length-stratified results saved to: {length_stats_path}")


if __name__ == "__main__":
    main()
