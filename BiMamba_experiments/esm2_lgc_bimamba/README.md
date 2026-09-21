# Sliding-window ESM2-3B + LGC + BiMamba 分支实验

## 实验目的

本目录在不修改基线代码的前提下，把原下游 Transformer 替换为 BiMamba，验证其能否在恢复后的完整蛋白序列上补充 sliding-window ESM2 缺失的跨窗口上下文。

```text
预先提取的全长 ESM2-3B residue embedding [L,2560]
  → 特征投影 [L,256]
  → 原项目 LGC（局部三维几何）
  → BiMamba × 2（全长双向上下文）
  → MLP
  → 每个残基的 RNA-binding logit
```

与基线的核心差异只有全局序列模块：`Transformer → BiMamba`。数据、LGC、Focal Loss、AdamW、随机种子、训练轮数、batch size、梯度累积和 early stopping 均沿用当前基线。模型不会运行或重新提取 ESM2。

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | 集中设置数据路径、模型、训练、推理和输出参数 |
| `dataset_adapter.py` | 严格读取已有 `.pt`，校验长度且绝不截断 |
| `bimamba_block.py` | Forward/Backward Mamba、Concat+Linear 融合和 FFN |
| `model_bimamba.py` | 复用原 SDE/LGC 的完整分支模型 |
| `train_bimamba.py` | 只训练和验证，保存 checkpoint、验证预测及指标 |
| `infer_bimamba.py` | 只加载 checkpoint 并执行测试集预测与分析 |
| `utils_bimamba.py` | 指标、阈值、随机种子和文件输出工具 |
| `tests/` | 形状、反转、反向传播和 500～2500 aa 长序列测试 |

## 数据输入

每个 `.pt` 文件表示一个蛋白，必需字段为：

```text
esm2_embedding: [L,2560]
ca_coords:      [L,3]
label:          [L]
coord_mask:     [L]（可选）
id/sequence/length（可选元数据）
```

如果文件中包含 `sequence` 或 `length`，加载器会把它们与 embedding、坐标、标签长度一起校验；任何不一致都会报告 sample_id 和各字段长度并立即报错。代码没有 `max_len` 参数，也没有 1022 截断逻辑。

## BiMamba 定义

```text
H_f = Mamba_f(LN(X))
H_b = Reverse(Mamba_b(Reverse(LN(X))))
H   = W_o [H_f || H_b] + b_o
X'  = X + Dropout(H)
X_out = X' + FFN(LN(X'))
```

batch 内不同长度的蛋白先经带 mask 的 LGC，再按真实长度逐条切片进入 BiMamba，最后补齐。因此反向分支不会把 padding 与真实残基混合。

## 依赖

依赖版本记录在 `requirements_bimamba.txt`。默认后端是 `mamba_ssm.Mamba(d_state=16, d_conv=4, expand=2)`。`mamba-ssm` 通常需要匹配的 Linux、CUDA、PyTorch 和编译环境；若环境只支持 Mamba2，可在 `config.py` 中把 `mamba_backend` 改为 `"mamba2"`，结果中应同时记录这一差异。

## 直接运行

1. 打开 `config.py`，修改 `train_dir`、可选的 `val_dir` 和 `test_dir`。
2. 在 IDE 中直接运行 `train_bimamba.py`。该脚本不会读取测试目录。
3. 训练完成后，在 IDE 中直接运行 `infer_bimamba.py`。

两个入口都没有命令行参数。如果 `val_dir=None`，训练集按照 seed=42 固定进行 8:2 划分，具体文件清单保存到 `results/data_split.json`。

若要生成 Transformer 对比结果，在 `config.py` 中填写 `INFERENCE.baseline_checkpoint` 对应的默认值，然后重新运行推理文件。基线也只使用其 checkpoint 保存的验证集阈值，绝不会在测试集调阈值。

## 训练输出

```text
checkpoints/esm2_lgc_bimamba_best.pt  最佳模型、优化器、阈值、配置和划分信息
logs/train_bimamba.log                训练日志
logs/train_history.csv                每轮 loss、指标、耗时和显存
results/data_split.json               可复现的数据划分
results/validation_predictions.csv    验证集逐残基概率与预测
results/validation_metrics.json       最佳权重的验证指标
results/training_summary.json         参数量、配置与最佳 epoch
```

训练脚本按验证集 MCC 选择模型，并在 `[0.01,0.99]` 搜索最佳阈值。

## 推理输出

```text
results/test_predictions.csv          BiMamba 测试集逐残基预测
results/test_metrics.json             测试集整体指标
results/summary.csv                    论文汇总字段
results/length_group_metrics.csv       <500、500-1000、1000-1500、1500-2000、>=2000
results/runtime_by_length.csv          每个蛋白的推理时间和增量峰值显存
```

填写基线 checkpoint 后还会生成 `baseline_test_predictions.csv` 和 `comparison_deltas.csv`，并把两种方法同时写入三个汇总表。

## 默认参数

```text
d_model=256, num_bimamba_layers=2
d_state=16, d_conv=4, expand=2, dropout=0.1
classifier_hidden_dim=128, k_neighbors=16, cutoff=30 Å
```

需要做 1/2/4 层消融时，只修改 `config.py` 的 `num_bimamba_layers`，并为不同实验设置不同输出目录或 checkpoint 名称，避免覆盖结果。

## 长序列说明

BiMamba 本身对序列长度是线性的，但当前复用的 LGC/SDE 会显式构造 `[L,L]` 距离矩阵，因此完整模型的显存仍可能受 LGC 的二次复杂度限制。若发生 OOM，代码会报出 sample_id 和长度，不会截断或静默跳过；应优先减小 batch size、增加梯度累积或使用更大显存设备。

