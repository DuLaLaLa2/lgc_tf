import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [p for p in sys.path if p != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from fution_try.model_a_residual_fusion import LGCTransformerResidualFusion  # noqa: E402
from fution_try.model_b_concat_fusion import LGCTransformerConcatFusion  # noqa: E402
from fution_try.model_c_gate_fusion import LGCTransformerGateFusion  # noqa: E402
from src.data.rna_dataset import RNABindingDataset, pad_collate_fn  # noqa: E402
from src.losses import BinaryFocalLossWithLogits  # noqa: E402
from src.utils.metrics import compute_binary_metrics, find_best_threshold  # noqa: E402


DATA_ROOT = Path("/root/autodl-tmp/features/merged")
TRAIN_DIR = DATA_ROOT / "RNA_Train"
VAL_DIR = DATA_ROOT / "RNA_Valid"
TEST_DIR = DATA_ROOT / "RNA-157_Test"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

VAL_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCHEME_SPECS = {
    "A": {
        "name": "A_residual_fusion",
        "description": "h_final = h_tf + h_lgc",
        "model_cls": LGCTransformerResidualFusion,
    },
    "B": {
        "name": "B_concat_fusion",
        "description": "h_final = MLP(concat[h_lgc, h_tf])",
        "model_cls": LGCTransformerConcatFusion,
    },
    "C": {
        "name": "C_gate_fusion",
        "description": "h_final = gate * h_tf + (1 - gate) * h_lgc",
        "model_cls": LGCTransformerGateFusion,
    },
}


def amp_enabled(args: argparse.Namespace) -> bool:
    return args.amp and DEVICE.type == "cuda"


def set_seed(seed: int) -> None:
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


def build_model(scheme: str, args: argparse.Namespace):
    model_cls = SCHEME_SPECS[scheme]["model_cls"]
    return model_cls(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        sde_dim=args.sde_dim,
        cutoff=args.cutoff,
        k_neighbors=args.k_neighbors,
        use_seq_pe=args.seq_pe,
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


def collect_predictions(model, loader, args: argparse.Namespace):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=DEVICE.type, enabled=amp_enabled(args)):
                logits, y, mask = forward_batch(model, batch)

            all_probs.append(torch.sigmoid(logits[mask]).detach().cpu().numpy())
            all_labels.append(y[mask].detach().cpu().numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def evaluate(model, loader, args: argparse.Namespace, threshold=None):
    y_true, y_prob = collect_predictions(model, loader, args)
    if threshold is None:
        threshold, metrics = find_best_threshold(y_true=y_true, y_prob=y_prob, metric_name="mcc")
    else:
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    return metrics, threshold


def build_datasets(args: argparse.Namespace):
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
            seed=args.seed,
        )
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        split_protocol = f"train split fallback: train/val = 8/2, seed={args.seed}"

    return full_train_dataset, train_dataset, val_dataset, test_dataset, split_protocol


def metric_value(metrics: dict, key: str):
    value = metrics.get(key, "")
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_result(result_file: Path, row: dict) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scheme",
        "method",
        "description",
        "val_mcc",
        "test_mcc",
        "test_auc",
        "test_aupr",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_accuracy",
        "best_threshold",
        "best_epoch",
        "seed",
        "checkpoint",
        "split_protocol",
    ]
    write_header = not result_file.exists()

    with result_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one_scheme(scheme: str, args: argparse.Namespace) -> dict:
    spec = SCHEME_SPECS[scheme]
    set_seed(args.seed)

    full_train_dataset, train_dataset, val_dataset, test_dataset, split_protocol = build_datasets(args)
    alpha, pos_ratio, pos_count, neg_count = estimate_positive_alpha(full_train_dataset)

    print("=" * 80)
    print(f"Scheme {scheme}: {spec['name']}")
    print(f"Description : {spec['description']}")
    print(f"Train dir   : {args.train_dir}")
    print(f"Val dir     : {args.val_dir}")
    print(f"Test dir    : {args.test_dir}")
    print(f"Device      : {DEVICE}")
    print(f"Split       : {split_protocol}")
    print(f"Samples     : train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Pos ratio   : {pos_ratio:.6f} ({int(pos_count)} pos / {int(neg_count)} neg)")
    print("=" * 80)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "collate_fn": pad_collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_model(scheme, args).to(DEVICE)
    criterion = BinaryFocalLossWithLogits(alpha=alpha, gamma=args.focal_gamma, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=amp_enabled(args))

    ckpt_dir = args.output_dir / "checkpoints-157"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{spec['name']}_best.pt"

    best_val_mcc = -1e9
    best_epoch = -1
    best_threshold = 0.5
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            with autocast(device_type=DEVICE.type, enabled=amp_enabled(args)):
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
        val_metrics, val_threshold = evaluate(model=model, loader=val_loader, args=args, threshold=None)
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
                        "scheme": scheme,
                        "method": spec["name"],
                        "description": spec["description"],
                        "input_dim": args.input_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "ffn_dim": args.ffn_dim,
                        "dropout": args.dropout,
                        "sde_dim": args.sde_dim,
                        "cutoff": args.cutoff,
                        "k_neighbors": args.k_neighbors,
                        "seq_pe": args.seq_pe,
                        "batch_size": args.batch_size,
                        "accum_steps": args.accum_steps,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "focal_alpha": alpha,
                        "focal_gamma": args.focal_gamma,
                        "split_protocol": split_protocol,
                    },
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        print(
            f"[{scheme}] Epoch {epoch:03d} | "
            f"loss={avg_train_loss:.6f} | "
            f"val_mcc={val_metrics['mcc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_aupr={val_metrics['aupr']:.4f} | "
            f"thr={val_threshold:.2f} | "
            f"{'BEST' if improved else ''}"
        )

        if bad_epochs >= args.patience:
            print(f"[{scheme}] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print(f"[{scheme}] Loading best checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, _ = evaluate(model=model, loader=test_loader, args=args, threshold=best_threshold)

    row = {
        "scheme": scheme,
        "method": spec["name"],
        "description": spec["description"],
        "val_mcc": best_val_mcc,
        "test_mcc": metric_value(test_metrics, "mcc"),
        "test_auc": metric_value(test_metrics, "auc"),
        "test_aupr": metric_value(test_metrics, "aupr"),
        "test_f1": metric_value(test_metrics, "f1"),
        "test_precision": metric_value(test_metrics, "precision"),
        "test_recall": metric_value(test_metrics, "recall"),
        "test_specificity": metric_value(test_metrics, "specificity"),
        "test_accuracy": metric_value(test_metrics, "accuracy"),
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "split_protocol": split_protocol,
    }

    append_result(args.result_file, row)
    print(f"[{scheme}] Test MCC={row['test_mcc']:.6f}, AUC={row['test_auc']:.6f}, AUPR={row['test_aupr']:.6f}")
    print(f"[{scheme}] Result appended to: {args.result_file}")
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LGC + Transformer fusion branch experiments.")
    parser.add_argument("--scheme", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--result-file", type=Path, default=None)

    parser.add_argument("--input-dim", type=int, default=2560)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sde-dim", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=30.0)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--seq-pe", dest="seq_pe", action="store_true", default=True)
    parser.add_argument("--no-seq-pe", dest="seq_pe", action="store_false")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    args = parser.parse_args(argv)
    if args.result_file is None:
        args.result_file = args.output_dir / "fusion_comparison.csv"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schemes = ["A", "B", "C"] if args.scheme == "all" else [args.scheme]
    rows = []
    for scheme in schemes:
        rows.append(run_one_scheme(scheme, args))

    print("=" * 80)
    print("Fusion branch comparison")
    print("=" * 80)
    for row in rows:
        print(
            f"{row['scheme']} {row['method']}: "
            f"val_mcc={row['val_mcc']:.6f}, "
            f"test_mcc={row['test_mcc']:.6f}, "
            f"test_auc={row['test_auc']:.6f}, "
            f"ckpt={row['checkpoint']}"
        )
    print(f"All results file: {args.result_file}")


if __name__ == "__main__":
    main()
