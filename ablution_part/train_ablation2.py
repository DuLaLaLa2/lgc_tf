"""消融实验统一训练脚本。

用法示例：
    python -m ablation_part.train_ablation --variant no_transformer
    python -m ablation_part.train_ablation --variant only_esm2
    python -m ablation_part.train_ablation --variant esm2_lgc

每一个变体会产出：
    ablation_part/outputs/checkpoints/<tag>.pt          最优权重
    ablation_part/outputs/results/<tag>_test.json       配置及测试集总体指标
    ablation_part/outputs/results/<tag>_length.csv      按长度分 bin 的 MCC
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
if torch.cuda.is_available():
    torch.cuda.init()
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

# 支持从项目根目录以 -m 运行，也支持直接执行此文件。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.rna_dataset import RNABindingDataset, pad_collate_fn
from src.losses import BinaryFocalLossWithLogits
from src.utils.metrics import compute_binary_metrics, find_best_threshold

from ablution_part.models.ablation_model import AblationESM2Model


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/root/autodl-tmp/features/merged")

DEFAULT_TRAIN_DIR = DATA_ROOT / "RNA-495_Train"
DEFAULT_TEST_DIR = DATA_ROOT / "RNA_1346_New"
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
RESULT_DIR = PROJECT_DIR / "outputs" / "results"

SEED = 42
VAL_RATIO = 0.2

INPUT_DIM = 2560
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# tag, LGC, Seq-PE, additional Transformer, k
VARIANTS = {
    "full": ("A1_full", True, True, True, 16),
    "no_lgc": ("A2_no_lgc", False, True, True, 16),
    "no_pe": ("A3_no_pe", True, False, True, 16),
    "no_lgc_no_pe": ("A4_no_lgc_no_pe", False, False, True, 16),
    "k8": ("A5_k8", True, False, True, 8),
    "k32": ("A5_k32", True, False, True, 32),
    "no_transformer": ("A6_no_transformer", True, True, False, 16),
    "only_esm2": ("A7_only_esm2", False, False, False, 16),
    "esm2_lgc": ("A8_esm2_lgc", True, False, False, 16),
}


# 与主线实验保持一致的长度划分
LENGTH_BINS = (
    ("<500", lambda length: length < 500),
    ("500-999", lambda length: 500 <= length < 1000),
    ("1000-1499", lambda length: 1000 <= length < 1500),
    (">=1500", lambda length: length >= 1500),
)


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


def forward_batch(model, batch, use_amp: bool):
    x = batch["x"].to(DEVICE, non_blocking=True)
    y = batch["y"].to(DEVICE, non_blocking=True)
    mask = batch["mask"].to(DEVICE, non_blocking=True)
    coords = batch["coords"].to(DEVICE, non_blocking=True) if model.use_lgc else None
    coord_mask = batch["coord_mask"].to(DEVICE, non_blocking=True) if model.use_lgc else None
    logits = model(x=x, coords=coords, mask=mask, coord_mask=coord_mask)
    return logits, y, mask


def collect_predictions(model, loader, use_amp: bool):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                logits, y, mask = forward_batch(model, batch, use_amp)
            all_probs.append(torch.sigmoid(logits[mask]).detach().float().cpu().numpy())
            all_labels.append(y[mask].detach().cpu().numpy())
    return np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)


def evaluate(model, loader, use_amp: bool, threshold=None):
    y_true, y_prob = collect_predictions(model, loader, use_amp)
    if threshold is None:
        threshold, metrics = find_best_threshold(y_true=y_true, y_prob=y_prob, metric_name="mcc")
    else:
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    return metrics, threshold


def get_length_bin(length: int) -> str:
    for name, pred in LENGTH_BINS:
        if pred(length):
            return name
    raise ValueError(length)


def evaluate_by_length(model, loader, use_amp: bool, threshold: float):
    model.eval()
    grouped = {name: {"protein_count": 0, "labels": [], "probs": []} for name, _ in LENGTH_BINS}
    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                logits, y, mask = forward_batch(model, batch, use_amp)
            probabilities = torch.sigmoid(logits)
            for i in range(mask.shape[0]):
                sample_mask = mask[i].bool()
                L = int(sample_mask.sum().item())
                name = get_length_bin(L)
                grouped[name]["protein_count"] += 1
                grouped[name]["labels"].append(
                    y[i][sample_mask].detach().cpu().numpy().reshape(-1)
                )
                grouped[name]["probs"].append(
                    probabilities[i][sample_mask].detach().float().cpu().numpy().reshape(-1)
                )
    rows = []
    for name, _ in LENGTH_BINS:
        g = grouped[name]
        if g["protein_count"] == 0:
            rows.append({"length_bin": name, "protein_count": 0, "residue_count": 0, "mcc": float("nan")})
            continue
        y_true = np.concatenate(g["labels"], axis=0)
        y_prob = np.concatenate(g["probs"], axis=0)
        m = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
        rows.append({
            "length_bin": name,
            "protein_count": g["protein_count"],
            "residue_count": int(y_true.size),
            "mcc": float(m["mcc"]),
        })
    return rows


def save_length_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("length_bin", "protein_count", "residue_count", "mcc"))
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ESM2 / LGC / Transformer ablations")
    p.add_argument("--variant", choices=tuple(VARIANTS), help="named architecture preset")
    p.add_argument("--tag", type=str, help="output name; defaults to the preset tag")
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="check model/config without reading data or writing outputs")

    # Ablation knobs
    for name in ("lgc", "seq-pe", "transformer"):
        dest = "use_" + name.replace("-", "_")
        group = p.add_mutually_exclusive_group()
        group.add_argument(f"--use-{name}", dest=dest, action="store_true")
        group.add_argument(f"--no-{name}", dest=dest, action="store_false")
        p.set_defaults(**{dest: None})
    p.add_argument("--k-neighbors", type=int)

    args = p.parse_args(argv)
    if args.variant is None and args.tag is None:
        p.error("provide --variant or --tag for a custom configuration")
    preset = VARIANTS[args.variant] if args.variant else VARIANTS["full"]
    if args.tag is None:
        args.tag = preset[0]
    for name, value in zip(("use_lgc", "use_seq_pe", "use_transformer", "k_neighbors"), preset[1:]):
        explicit = getattr(args, name)
        if args.variant and explicit is not None and explicit != value:
            p.error(f"{name} conflicts with --variant {args.variant}; use --tag without --variant for a custom configuration")
        if explicit is None:
            setattr(args, name, value)
    for name in ("epochs", "patience", "batch_size", "accum_steps", "k_neighbors"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    set_seed(SEED)

    use_amp = args.amp and DEVICE.type == "cuda"

    model_config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "ffn_dim": FFN_DIM,
        "dropout": DROPOUT,
        "sde_dim": 16,
        "cutoff": 30.0,
        "k_neighbors": args.k_neighbors,
        "use_lgc": args.use_lgc,
        "use_seq_pe": args.use_seq_pe,
        "use_transformer": args.use_transformer,
    }
    training_config = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_workers": args.num_workers,
        "amp": use_amp,
        "focal_gamma": 2.0,
    }
    if args.dry_run:
        model = AblationESM2Model(**model_config)
        print(json.dumps({
            "tag": args.tag,
            "variant": args.variant,
            "model_config": model_config,
            "training_config": training_config,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }, indent=2, ensure_ascii=False))
        return

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Ablation run: {args.tag}")
    print(f"  use_lgc       = {args.use_lgc}")
    print(f"  use_seq_pe    = {args.use_seq_pe}")
    print(f"  use_transformer={args.use_transformer}")
    print(f"  k_neighbors   = {args.k_neighbors}")
    print(f"  DEVICE        = {DEVICE}")
    print("=" * 80)

    full_train = RNABindingDataset(args.train_dir, max_len=None)
    test_ds = RNABindingDataset(args.test_dir, max_len=None)

    tr_idx, va_idx = split_train_val(len(full_train), VAL_RATIO, SEED)
    train_ds = Subset(full_train, tr_idx)
    val_ds = Subset(full_train, va_idx)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    alpha, pos_ratio, pos_c, neg_c = estimate_positive_alpha(full_train)
    print(f"pos_ratio={pos_ratio:.4f}  alpha={alpha:.4f}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=pad_collate_fn,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = AblationESM2Model(**model_config).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_config["focal_alpha"] = alpha
    print(f"trainable_parameters={parameter_count:,}")

    criterion = BinaryFocalLossWithLogits(alpha=alpha, gamma=2.0, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_mcc = -1e9
    best_epoch = -1
    best_threshold = 0.5
    bad = 0
    ckpt_path = CKPT_DIR / f"{args.tag}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                logits, y, mask = forward_batch(model, batch, use_amp)
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

        val_metrics, val_thr = evaluate(model, val_loader, use_amp, threshold=None)
        improved = val_metrics["mcc"] > best_val_mcc
        if improved:
            best_val_mcc = val_metrics["mcc"]
            best_epoch = epoch
            best_threshold = val_thr
            bad = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_mcc": best_val_mcc,
                "best_threshold": best_threshold,
                "config": {
                    **model_config,
                    "focal_alpha": alpha,
                    "tag": args.tag,
                },
                "variant": args.variant,
                "training_config": training_config,
                "trainable_parameters": parameter_count,
            }, ckpt_path)
        else:
            bad += 1

        avg_loss = total_loss / max(total_steps, 1)
        print(
            f"[{args.tag}] ep{epoch:02d} loss={avg_loss:.4f} "
            f"val_mcc={val_metrics['mcc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} thr={val_thr:.2f} "
            f"{'BEST' if improved else ''}"
        )
        if bad >= args.patience:
            print(f"[{args.tag}] early stop at ep{epoch}, best ep{best_epoch}")
            break

    print(f"[{args.tag}] loading best ckpt...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, _ = evaluate(model, test_loader, use_amp, threshold=best_threshold)
    length_rows = evaluate_by_length(model, test_loader, use_amp, best_threshold)

    print(f"[{args.tag}] test MCC = {test_metrics['mcc']:.4f}")
    for row in length_rows:
        print(f"  {row['length_bin']:>12} N={row['protein_count']:4d}  R={row['residue_count']:7d}  MCC={row['mcc']:.4f}")

    # 保存
    overall_path = RESULT_DIR / f"{args.tag}_test.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "variant": args.variant,
            "model_config": model_config,
            "training_config": training_config,
            "trainable_parameters": parameter_count,
            "best_val_mcc": best_val_mcc,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "test_overall": {k: (float(v) if isinstance(v, (float, int, np.floating)) else v)
                             for k, v in test_metrics.items()},
        }, f, indent=2, ensure_ascii=False)

    save_length_csv(length_rows, RESULT_DIR / f"{args.tag}_length.csv")


if __name__ == "__main__":
    main()
