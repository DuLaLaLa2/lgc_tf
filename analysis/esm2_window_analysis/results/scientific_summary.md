# ESM2 window-count post-hoc analysis

本分析仅重新聚合既有测试集逐残基预测，没有重新训练、重新提取 ESM2、改变测试集或重新选择阈值。

- Transformer validation-derived threshold: 0.58
- BiMamba validation-derived threshold: 0.61
- Sliding window: window_size=1022, overlap=122, stride=900

## 核心结果

| window_group | protein_count | residue_count | positive_ratio | transformer_mcc | transformer_macro_mcc | transformer_aupr | transformer_macro_aupr | transformer_f1 | transformer_macro_f1 | transformer_precision | transformer_recall | transformer_auc | transformer_specificity | transformer_accuracy | bimamba_mcc | bimamba_macro_mcc | bimamba_aupr | bimamba_macro_aupr | bimamba_f1 | bimamba_macro_f1 | bimamba_precision | bimamba_recall | bimamba_auc | bimamba_specificity | bimamba_accuracy | delta_mcc | delta_macro_mcc | delta_aupr | delta_macro_aupr | delta_f1 | delta_macro_f1 | delta_precision | delta_recall | delta_auc | delta_specificity | delta_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-window | 1295 | 350783 | 0.0719191 | 0.366188 | 0.292618 | 0.370908 | 0.39004 | 0.413016 | 0.340881 | 0.346135 | 0.511931 | 0.84611 | 0.92506 | 0.895348 | 0.380845 | 0.298978 | 0.392007 | 0.396788 | 0.422871 | 0.355036 | 0.33707 | 0.567267 | 0.863769 | 0.913545 | 0.888641 | 0.0146575 | 0.00635986 | 0.0210988 | 0.00674827 | 0.00985498 | 0.0141545 | -0.00906484 | 0.0553353 | 0.017659 | -0.0115157 | -0.00670785 |
| 2-window | 43 | 54478 | 0.0215683 | 0.171143 | 0.140599 | 0.138067 | 0.173785 | 0.189927 | 0.13139 | 0.173212 | 0.210213 | 0.827977 | 0.977881 | 0.961324 | 0.213399 | 0.162175 | 0.156053 | 0.145558 | 0.225924 | 0.157529 | 0.175364 | 0.317447 | 0.857488 | 0.967094 | 0.953082 | 0.0422563 | 0.0215757 | 0.0179867 | -0.0282266 | 0.0359967 | 0.0261384 | 0.00215258 | 0.107234 | 0.0295109 | -0.0107874 | -0.00824186 |
| 3-window | 8 | 16928 | 0.0199669 | 0.183849 | 0.178635 | 0.128401 | 0.156403 | 0.200825 | 0.181135 | 0.187661 | 0.215976 | 0.83021 | 0.980952 | 0.965678 | 0.146474 | 0.129144 | 0.108977 | 0.121751 | 0.164216 | 0.136665 | 0.140167 | 0.198225 | 0.816104 | 0.975226 | 0.959712 | -0.0373747 | -0.0494908 | -0.0194232 | -0.0346515 | -0.0366096 | -0.0444696 | -0.0474933 | -0.0177515 | -0.0141058 | -0.00572634 | -0.00596645 |

## Protein-level bootstrap（ΔMCC）

| window_group | protein_count | observed_delta_mcc | bootstrap_mean_delta_mcc | ci_2.5 | ci_97.5 | n_bootstrap | bootstrap_unit | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-window | 1295 | NaN | NaN | NaN | NaN | 2000 | protein | 42 |
| 2-window | 43 | 0.0422563 | 0.0408327 | 0.0022653 | 0.0792373 | 2000 | protein | 42 |
| 3-window | 8 | -0.0373747 | -0.0383774 | -0.0912687 | 0.00157065 | 2000 | protein | 42 |

## Spearman 相关

| variable | spearman_rho | p_value | valid_protein_count |
| --- | --- | --- | --- |
| esm2_window_count | -0.00340096 | 0.900794 | 1346 |
| sequence_length | 0.0913542 | 0.000792223 | 1346 |
| positive_ratio | -0.0500224 | 0.0665564 | 1346 |

## 科学解释边界

- 1-window 的 ΔMCC=0.014658。
- 2-window 的 ΔMCC=0.042256。
- 3-window 的 ΔMCC=-0.037375。
- 是否随 window_count 系统变化，应同时结合 macro 指标、蛋白级 bootstrap 区间和 Spearman 结果判断。
- positive_ratio 若随 window_count 改变，则类别不平衡与窗口数量存在混杂，不能把 pooled 指标变化单独归因于跨窗口上下文。
- 即使多窗口组增益更大，这一 post-hoc 结果也只能支持关联性证据，不能自动证明 BiMamba 已恢复 ESM2 跨窗口信息。
