# ESM2 + Parallel LGC/BiMamba 分支实验

## 唯一结构差异

本分支与前一个串联分支使用相同数据、随机种子、LGC、BiMamba、Focal Loss、优化器、验证策略和测试分析。唯一核心变化是 LGC 与 BiMamba 并行：

```text
                         ┌─ LGC ────────> H_lgc [B,L,256] ──────┐
ESM2 [B,L,2560] -> 投影 ─┤                                      ├─ Concat
                         └─ BiMamba × 2 -> H_bimamba [B,L,256] ─┘
                                         H_final [B,L,512]
                                                ↓
                                               MLP
                                                ↓
                                         logits [B,L]
```

`H_lgc` 和 `H_bimamba` 均以同一份投影后 ESM2 特征为输入。`H_final = Concat(H_lgc, H_bimamba)`，拼接后不经过 `2d→d` 压缩，分类器直接接收 512 维融合特征。这一点与串联分支“LGC 输出作为 BiMamba 输入”不同。

## 运行方式

1. 修改 `config.py` 中的训练、验证和测试路径。
2. 在 IDE 中直接运行 `train_parallel.py`；训练脚本不读取测试集。
3. 训练完成后直接运行 `infer_parallel.py`；推理固定使用 checkpoint 中的验证阈值。

两个入口均无命令行参数，不会提取 ESM2，也不会截断长蛋白。

## 输入格式

每个 `.pt` 文件至少包含：

```text
esm2_embedding [L,2560]
ca_coords      [L,3]
label          [L]
coord_mask     [L]（可选）
```

存在 sequence/length 元数据时也会参与严格长度校验。

## 输出

训练输出：

```text
checkpoints/esm2_lgc_bimamba_parallel_best.pt
logs/train_parallel.log
logs/train_history.csv
results/data_split.json
results/validation_predictions.csv
results/validation_metrics.json
results/training_summary.json
```

推理输出：

```text
results/test_predictions.csv
results/test_metrics.json
results/summary.csv
results/length_group_metrics.csv
results/runtime_by_length.csv
```

在 `config.py` 填写可选的 Transformer 基线 checkpoint 后，推理还会输出基线预测和 `comparison_deltas.csv`。

`tests/test_parallel_model.py` 检查并行分支张量维度；直接运行
`tests/test_long_sequence.py` 会测试 500、1000、1500、2000、2500 aa，并把
BiMamba 核心的时间和显存写入 `results/bimamba_long_sequence_benchmark.csv`。

## 默认参数与长序列

默认 `d_model=256`、BiMamba 层数 2、`d_state=16`、`d_conv=4`、`expand=2`、dropout 0.1。BiMamba 会按每条蛋白真实长度运行，padding 不参与 reverse、loss 或指标。完整模型仍复用构造 `[L,L]` 距离矩阵的 LGC，因此长序列显存可能主要受 LGC 的二次复杂度约束；OOM 会显式报告样本 ID 和长度，不会截断或跳过。
