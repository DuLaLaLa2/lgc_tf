# LGC + Transformer Fusion Branch Experiments

This folder contains three independent fusion variants for comparing LGC output
features and Transformer output features.

## Schemes

- A: residual fusion, `h_final = h_tf + h_lgc`
- B: concat fusion, `h_final = MLP(concat[h_lgc, h_tf])`
- C: gated fusion, `h_final = gate * h_tf + (1 - gate) * h_lgc`

Each scheme has its own model file:

- `model_a_residual_fusion.py`
- `model_b_concat_fusion.py`
- `model_c_gate_fusion.py`

## Default Data Layout

The default feature root is:

```bash
/root/autodl_tmp/features/merged
```

Expected folders:

```bash
/root/autodl_tmp/features/merged/RNA_Train
/root/autodl_tmp/features/merged/RNA_Valid
/root/autodl_tmp/features/merged/RNA_Test
```

If `RNA_Valid` does not exist or contains no `.pt` files, the script falls back
to an 8:2 train/validation split from `RNA_Train` with seed 42.

## Run

Run one scheme:

```bash
python fution_try/train_fution.py --scheme A
python fution_try/train_fution.py --scheme B
python fution_try/train_fution.py --scheme C
```

Run all schemes serially:

```bash
python fution_try/train_fution.py --scheme all
```

## Outputs

Best checkpoints are saved separately:

```bash
fution_try/outputs/checkpoints/A_residual_fusion_best.pt
fution_try/outputs/checkpoints/B_concat_fusion_best.pt
fution_try/outputs/checkpoints/C_gate_fusion_best.pt
```

Comparison metrics are appended to one CSV:

```bash
fution_try/outputs/fusion_comparison.csv
```
