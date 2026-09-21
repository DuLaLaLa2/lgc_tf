# ESM2 sliding-window 分组测试结果分析

本目录只对已有测试集逐残基预测做 post-hoc subgroup analysis。分析不会：

- 重新训练 Transformer、BiMamba、LGC 或 ESM2；
- 重新提取 ESM2 特征；
- 改变 train / validation / test split、标签或预测概率；
- 在测试集或窗口子组上重新选择 threshold；
- 删除长蛋白、少样本组或异常结果。

唯一变化是把已有测试蛋白从普通长度区间改为按实际 ESM2 window count 分组，再重新聚合原始 residue-level predictions。

## 已确认的原始滑窗逻辑

原始代码位于：

`C:\Code\Git_clone_resp\3D-SRPE-main\3D-SRPE-main\extract_esm2_from_rna_dataset.py`

脚本中的真实参数和循环为：

- `window_size = 1022`
- `stride = 900`
- `overlap = 122`
- `end = min(start + 1022, sequence_length)`
- 若 `end == sequence_length` 则结束，否则 `start += 900`

分析脚本复制了该循环来计算 window count，并在每次正式分析前自动执行任务指定的 13 个边界测试。分组由 window count 生成，不硬编码长度区间；`1-window`、`2-window`、`3-window` 分别报告，四个及以上窗口合并为 `>=4-window`。不存在的组不会生成。

## 输入

需要两份同一正式测试集的逐残基 CSV：

- Transformer：通常为 `baseline_test_predictions.csv`
- BiMamba：通常为 `test_predictions.csv`

默认兼容项目推理脚本的字段：

`sample_id, protein_length, residue_index, label, probability, threshold`

也兼容任务描述中的常见别名。每份 CSV 必须包含 residue index，以便严格验证逐残基对应关系。threshold 必须是全文件唯一的正式 validation-derived threshold；若 CSV 没有 threshold 列，必须通过命令行明确传入，脚本不会从测试集估计阈值。

## 运行

```powershell
python analysis/esm2_window_analysis/analyze_window_groups.py `
  --transformer-predictions ./src/outputs/predictions/baseline_test_predictions.csv `
  --bimamba-predictions ./BiMamba_experiments/esm2_lgc_bimamba/results/test_predictions.csv `
  --output-dir analysis/esm2_window_analysis/results `
  --n-bootstrap 2000 `
  --seed 42
```

如果预测 CSV 不含 threshold 列，再增加：

```text
--transformer-threshold <验证集阈值> --bimamba-threshold <验证集阈值>
```

脚本拒绝覆盖已有结果。重复正式运行时应指定新的输出目录，以保留审计轨迹。

## 强制一致性检查

分析开始前会逐蛋白、逐残基检查：

- Transformer 与 BiMamba 的 protein ID 集合完全相同；
- 每个蛋白的 sequence length 相同，且等于逐残基行数；
- residue index 连续且两模型完全对应；
- window count 相同；
- label 逐残基完全相同；
- probability 有限且位于 `[0,1]`；
- 每个模型只使用自己的固定验证集阈值。

任何不一致都会打印相关 protein ID 并报错，不会静默取交集或跳过。

## 输出

结果目录包含：

- `window_analysis_protein_metadata.csv`：逐蛋白长度、window count、正负残基数和正类比例；
- `protein_metadata.csv`：同一元数据的兼容文件名；
- `transformer_by_esm2_window.csv`；
- `bimamba_by_esm2_window.csv`；
- `transformer_vs_bimamba_by_window.csv`；
- `per_protein_transformer_vs_bimamba.csv`；
- `window_bootstrap_delta_mcc.csv`：以蛋白为采样单位的 2000 次 bootstrap；
- `correlation_analysis.csv`：window count、长度、positive ratio 与逐蛋白 ΔMCC 的 Spearman 相关；
- `plot_window_metrics_long.csv` 和 `plot_sequence_length_delta_mcc.csv`：后续绘图数据；
- `analysis_warnings.log`：单类别蛋白等警告；
- `scientific_summary.md`：描述性结果与解释边界。

组内所有残基合并计算的结果为 micro evaluation；逐蛋白有效指标的均值为 macro evaluation。单蛋白真实标签只有一个类别时，MCC、AUC 和 AUPR 保存为 `NaN`，不会强制填为 0。

bootstrap 以蛋白为单位有放回抽样，再汇总抽中蛋白的混淆矩阵计算两模型 MCC 和 `BiMamba - Transformer`。因此不会把同一蛋白内的残基当作独立 bootstrap 样本。

## 当前数据状态

当前工作机尚未发现本次正式实验的两份逐残基预测 CSV，也未发现对应 checkpoint。项目配置指向服务器路径 `/root/autodl-tmp/features/merged/RNA_1346_New`，该目录不在本机。本机存在的 `RNA-117_Test` 是较旧数据，不能替代本次正式测试集。因此此目录目前只包含已验证的分析代码，没有伪造正式统计结果。
