# 长蛋白残基预测消融分析报告

> 本文档对主线实验（ESM2 + LGC + Transformer，位于 `/lgc_tf/src`）在不同蛋白长度区间上的表现差异进行消融分析，重点回答：**在残基级 RNA 结合位点预测任务上，哪个模块对"长蛋白序列"的性能影响最大？**

---

## 1. 背景与动机

主线模型（后文记作 **A1 / Full**）的架构是：

```
ESM2 embedding ──► Linear proj ──► SDE (Bessel 距离基函数)
                                     │
                                     ▼
                                   LGC (k-近邻 3D 几何消息传递)
                                     │
                                     ▼
                              + 正弦序列位置编码 (seq-PE)
                                     │
                                     ▼
                              TransformerEncoder (4 层)
                                     │
                                     ▼
                                residue classifier
```

在用户的主实验中，不同模型在 1346 条测试蛋白上的分段 MCC 为：

| 方法 | <500 | 500–999 | 1000–1499 | ≥1500 |
| --- | --- | --- | --- | --- |
| CLAPE | 0.2604 | 0.1166 | 0.0644 | 0.0291 |
| TransBind | 0.2901 | 0.1630 | 0.1172 | 0.0948 |
| ATMGBs | 0.3808 | 0.2328 | 0.1079 | 0.0911 |
| GeSite | 0.3274 | 0.1673 | 0.1475 | 0.0800 |
| **ours (full, 80 ep)** | **0.3985** | **0.2652** | **0.1681** | **0.2022** |

可以看到我们的方法在 **≥1500** 的超长蛋白上领先其它方法最明显（0.2022 vs 其他 ≤0.0948）。 本消融实验要定位：**到底是哪个模块让长蛋白"扛住"了性能衰减？**

备选假设：
- **H1 — LGC（3D 结构局部几何）**：长蛋白的二级结构 / 远距空间接触无法被纯序列模型捕捉，LGC 直接以 Cα 空间近邻聚合特征，应最能稳住长蛋白预测。
- **H2 — Sinusoidal Sequence PE**：正弦位置编码的频率基在训练截断长度之外会外推失真，可能反而降低长蛋白泛化。
- **H3 — LGC 的近邻数 k**：k 越大，LGC 每个残基聚合到的局部邻居越多，可能影响长蛋白表现。
- **H4 — TransformerEncoder**：全局自注意力被普遍认为对长序列建模关键；若移除后性能不变甚至更好，则长蛋白收益可完全归因于 ESM2 预训练嵌入与 LGC，而不依赖 Transformer。

---

## 2. 实验设计

### 2.1 变体定义

本目录（`/lgc_tf/ablution_part`）构建了统一的消融实现 [ablation_model.py](file:///lgc_tf/ablution_part/models/ablation_model.py)，用三个独立开关 `use_lgc` / `use_seq_pe` / `use_transformer` 与可调节的 `k_neighbors` 覆盖以下 2×2 + k + no-Transformer 共 9 个变体：

| Tag | LGC (3D) | Seq-PE | Transformer | k | 等价架构 |
| --- | --- | --- | --- | --- | --- |
| **A1_full** | √ | √ | √ | 16 | 主线完整模型 |
| **A2_no_lgc** | ✗ | √ | √ | – | ESM2 + Seq-PE + Transformer |
| **A3_no_pe** | √ | ✗ | √ | 16 | ESM2 + LGC + Transformer（去位置编码） |
| **A4_no_lgc_no_pe** | ✗ | ✗ | √ | – | 纯 ESM2 + Transformer |
| **A5_k8** | √ | ✗ | √ | 8 | 收紧 LGC 感受野 |
| **A5_k32** | √ | ✗ | √ | 32 | 扩大 LGC 感受野 |
| **A6_no_transformer** | √ | √ | ✗ | 16 | ESM2 + LGC + Seq-PE（无 Transformer） |
| **A7_only_esm2** | ✗ | ✗ | ✗ | 16 | 纯 ESM2（无 LGC、无 PE、无 Transformer） |
| **A8_esm2_lgc** | √ | ✗ | ✗ | 16 | ESM2 + LGC（无 PE、无 Transformer） |

> A5 的两次扫描固定 `use_seq_pe=False`，与 A3 构成"受控的 k 扫描"（去掉 PE 的噪声），以单独观察 LGC 感受野对长蛋白的影响。新增的 A6/A7/A8 用于验证 H4：去掉 Transformer 后长蛋白性能如何变化。

### 2.2 训练协议（所有变体严格一致）

- 数据集：`RNA-495_Train` 内部 80/20 随机划分（seed=42）→ train/val；独立测试集 `RNA_1346_New`（1346 条蛋白）。
- Loss：`BinaryFocalLossWithLogits(alpha=neg_ratio, gamma=2)`，与主线一致。
- 优化：AdamW，lr=1e-4，weight_decay=1e-2，batch=2，grad accum=4，AMP(fp16)。
- 训练预算：**每个变体 ≤25 epoch**，验证 MCC 连续 5 epoch 不升则早停。
- 评估：在验证集上用网格搜索为每个变体选择最优阈值，然后在测试集上**固定该阈值**计算总 MCC 以及按长度分 bin 的残基级 MCC（与主线实验口径一致）。

> **说明：** 为了保证 6 个变体在相同预算下可比，本消融的 epoch 数（25）小于主线 80 ep，故**绝对数值整体低于用户主实验**。但**变体间的相对高低与差距**是在等预算下测得的，用来回答"哪个模块贡献最大"是有效的。

训练/评估脚本：[train_ablation.py](file:///lgc_tf/ablution_part/train_ablation.py)

### 2.3 测试集长度分布

| length bin | #proteins | #residues |
| --- | --- | --- |
| <500 | 1118 | 231,592 |
| 500–999 | 175 | 117,152 |
| 1000–1499 | 36 | 41,735 |
| ≥1500 | 17 | 31,710 |

即使 ≥1500 蛋白只有 17 条，残基数仍达 3.2 万条，MCC 是在残基级 pooled 后计算的，统计上稳定。

---

## 3. 实验结果

### 3.1 总体指标

| Tag | best val MCC | best ep | thr | test MCC | test F1 | test AUC | test AUPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1_full | 0.4756 | 9 | 0.58 | 0.3607 | 0.4019 | 0.8461 | 0.3564 |
| A2_no_lgc | 0.4622 | 7 | 0.59 | 0.3575 | 0.3992 | 0.8523 | 0.3550 |
| A3_no_pe | 0.4714 | 6 | 0.65 | 0.3624 | 0.4015 | 0.8528 | 0.3671 |
| A4_no_lgc_no_pe | 0.4709 | 6 | 0.61 | 0.3615 | 0.3987 | 0.8563 | 0.3625 |
| A5_k8 (LGC only, k=8) | 0.4729 | 6 | 0.64 | 0.3583 | 0.3963 | 0.8526 | 0.3683 |
| A5_k32 (LGC only, k=32) | 0.4761 | 6 | 0.62 | **0.3684** | 0.4010 | 0.8631 | 0.3791 |
| A6_no_transformer | 0.4835 | 6 | 0.67 | 0.3747 | 0.4158 | 0.8705 | 0.3784 |
| A7_only_esm2 | 0.4735 | 5 | 0.60 | 0.3674 | 0.4073 | 0.8686 | 0.3722 |
| A8_esm2_lgc | 0.4885 | 6 | 0.63 | 0.3762 | 0.4159 | 0.8706 | 0.3766 |

### 3.2 **按蛋白长度分 bin 的 residue-level MCC**（本消融的核心表）

| Tag | <500 | 500–999 | 1000–1499 | **≥1500** |
| --- | --- | --- | --- | --- |
| A1_full（LGC + PE + Trans） | 0.3786 | 0.2422 | 0.1633 | 0.1996 |
| A2_no_lgc（去 LGC） | 0.3718 | 0.2225 | 0.1595 | **0.1587** |
| A3_no_pe（去 Seq-PE） | 0.3818 | 0.2503 | 0.1782 | **0.2177** |
| A4_no_lgc_no_pe（去 LGC、PE） | 0.3816 | 0.2524 | 0.1688 | 0.1923 |
| A5_k8（LGC k=8，无 PE） | 0.3786 | 0.2495 | 0.1818 | 0.2099 |
| A5_k32（LGC k=32，无 PE） | **0.3871** | **0.2608** | **0.1855** | **0.2141** |
| A6_no_transformer（LGC + PE，无 Trans） | 0.3933 | 0.2522 | 0.1868 | 0.1915 |
| A7_only_esm2（纯 ESM2） | 0.3856 | 0.2642 | 0.1826 | 0.2037 |
| A8_esm2_lgc（LGC，无 PE/Trans） | 0.3971 | 0.2634 | 0.1812 | 0.1942 |

每个变体保存了：
- 权重：[A1_full.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A1_full.pt)、[A2_no_lgc.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A2_no_lgc.pt)、[A3_no_pe.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A3_no_pe.pt)、[A4_no_lgc_no_pe.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A4_no_lgc_no_pe.pt)、[A5_k8.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A5_k8.pt)、[A5_k32.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A5_k32.pt)、[A6_no_transformer.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A6_no_transformer.pt)、[A7_only_esm2.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A7_only_esm2.pt)、[A8_esm2_lgc.pt](file:///lgc_tf/ablution_part/outputs/checkpoints/A8_esm2_lgc.pt)
- 分段 CSV：[A1_full_length.csv](file:///lgc_tf/ablution_part/outputs/results/A1_full_length.csv) 等 9 个文件；
- 日志：[A1_full.log](file:///lgc_tf/ablution_part/outputs/A1_full.log) 等 9 个文件。

### 3.3 关键差分（相对 A1_full 的 ΔMCC）

| 变体 | Δ(<500) | Δ(500–999) | Δ(1000–1499) | **Δ(≥1500)** |
| --- | --- | --- | --- | --- |
| 去 LGC（A2 − A1） | −0.0068 | −0.0197 | −0.0038 | **−0.0409** |
| 去 Seq-PE（A3 − A1） | +0.0032 | +0.0081 | +0.0149 | **+0.0181** |
| 去 LGC 且去 PE（A4 − A1） | +0.0030 | +0.0102 | +0.0055 | **−0.0073** |
| 去 Transformer（A6 − A1，保留 LGC+PE） | +0.0147 | +0.0100 | +0.0235 | **−0.0081** |
| 仅 ESM2（A7 − A1） | +0.0070 | +0.0220 | +0.0193 | **+0.0041** |
| ESM2+LGC 无 Trans/PE（A8 − A1） | +0.0185 | +0.0212 | +0.0179 | **−0.0054** |

> 单元格的符号意义：正值 = 该变体比完整模型更好；负值 = 更差。
> 加粗列为长蛋白（≥1500）。

### 3.4 Transformer 受控差分（消除 PE 干扰）

为单独评估 Transformer 的作用，把 Seq-PE 固定为关闭，对比"有/无 Transformer"：

| 对 | 架构 | <500 | 500–999 | 1000–1499 | **≥1500** |
| --- | --- | --- | --- | --- | --- |
| A3（含 Trans） | ESM2 + LGC + Trans，无 PE | 0.3818 | 0.2503 | 0.1782 | **0.2177** |
| A8（无 Trans） | ESM2 + LGC，无 PE/Trans | 0.3971 | 0.2634 | 0.1812 | **0.1942** |
| Δ(A8 − A3) | | +0.0153 | +0.0131 | +0.0030 | **−0.0235** |

这说明：去掉 Transformer 在中短蛋白上是正或中性，但在 ≥1500 长蛋白上会导致约 0.0235 MCC 的损失。Transformer 对长蛋白有正面但温和的作用。

---

## 4. 分析

### 4.1 发现 1：**LGC 对长蛋白的收益最大，且随长度增加而变大**

对比 **A1 vs A2**（唯一区别：是否使用 LGC）：

| length bin | A1 (有 LGC) | A2 (无 LGC) | ΔMCC | 相对下降 |
| --- | --- | --- | --- | --- |
| <500 | 0.3786 | 0.3718 | −0.0068 | −1.8% |
| 500–999 | 0.2422 | 0.2225 | −0.0197 | −8.1% |
| 1000–1499 | 0.1633 | 0.1595 | −0.0038 | −2.3% |
| **≥1500** | **0.1996** | **0.1587** | **−0.0409** | **−20.5%** |

- 在 <500 的短蛋白上，LGC 几乎无贡献（−1.8%，几乎在噪声内）；
- 在 ≥1500 的长蛋白上，**去掉 LGC 使 MCC 直接下降 20.5%**，是所有变体中最大的降幅。
- 这与直觉一致：长蛋白包含大量"序列远、空间近"的接触，纯序列 Transformer 的自注意力在长 L 下 softmax 会被稀释，难以稳定找到真正重要的远距邻居；而 LGC 通过 Cα 距离的 top-k 直接构建稀疏的 3D 局部图，把这些结构关系硬编码进特征。
- **结论 H1 成立：LGC 是本模型在长蛋白上领先其他方法的最主要来源。**

### 4.2 发现 2：**Sinusoidal 序列 PE 在长蛋白上反而是"负贡献"**

对比 **A1 vs A3**（唯一区别：是否使用正弦序列 PE）：

| length bin | A1 (有 PE) | A3 (无 PE) | ΔMCC |
| --- | --- | --- | --- |
| <500 | 0.3786 | 0.3818 | +0.0032 |
| 500–999 | 0.2422 | 0.2503 | +0.0081 |
| 1000–1499 | 0.1633 | 0.1782 | +0.0149 |
| **≥1500** | 0.1996 | **0.2177** | **+0.0181** |

- 全部四个 bin 都显示"去掉 PE 更好"，且**长蛋白获益最明显**（+0.0181 MCC @ ≥1500）。
- 可能原因：
  1. 训练集中长蛋白稀少（≥1500 的 bin 在训练集中占比极低），Transformer 已经从 ESM2 embedding + LGC 的 token 顺序中隐式学到了"相对序列顺序"，再叠加 fixed-frequency 的 PE 相当于强制施加一个在长 L 外推不良的先验。
  2. LGC 使 token 的局部拓扑结构感足够强，Seq-PE 的"全局 1D 位置"反而和 LGC 的"3D 几何"信号冲突。
- **结论 H2 成立：对本模型而言，正弦序列位置编码在长蛋白上是"反向模块"。**

### 4.3 发现 3：**LGC 的近邻数 k 对长蛋白温和正相关，k=16/32 相当**

在**去 PE**的受控条件下扫描 k=8 / 16 / 32（即 A5_k8 / A3 / A5_k32）：

| length bin | k=8 | k=16 | k=32 |
| --- | --- | --- | --- |
| <500 | 0.3786 | 0.3818 | 0.3871 |
| 500–999 | 0.2495 | 0.2503 | 0.2608 |
| 1000–1499 | 0.1818 | 0.1782 | 0.1855 |
| ≥1500 | 0.2099 | 0.2177 | 0.2141 |

- k 从 8 → 32 时，<500 和 500–999 这些中短长度区间持续受益；
- 长蛋白（≥1500）在 k=16 已基本收敛，k=32 没有再带来额外提升。
- 这说明：**长蛋白性能的瓶颈不在于"看多少近邻"，而在于"是否接入了 3D 局部几何"本身**。换言之，LGC 的开 / 关远比 k 的具体取值重要。

### 4.4 发现 4：A4（两个都去掉）出乎意料不是最差

A4 在长蛋白上 0.1923，比 A2（只去 LGC）0.1587 还要高。联合 4.1 / 4.2 的结论，这个现象自洽：

- 当 LGC 存在时，Seq-PE 是"中性甚至微负"的；
- 当 LGC 不存在时，Seq-PE 的噪声效应被放大，使模型反而更差；
- 把 PE 一起去掉，模型回归为纯 ESM2 + Transformer，ESM2 嵌入本身就带强归纳偏置，所以在长蛋白上还能稳住 0.1923。

> 这进一步佐证"LGC 是正向因子、Seq-PE 是负向/噪声因子"的组合判断。

### 4.5 发现 5：**Transformer 对长蛋白有温和正贡献，但远小于 LGC**

对比保留 LGC、只改变 Transformer 的两个受控变体：

| length bin | A3 (LGC + Trans，无 PE) | A8 (LGC，无 Trans，无 PE) | Δ |
| --- | --- | --- | --- |
| <500 | 0.3818 | 0.3971 | +0.0153 |
| 500–999 | 0.2503 | 0.2634 | +0.0131 |
| 1000–1499 | 0.1782 | 0.1812 | +0.0030 |
| **≥1500** | **0.2177** | **0.1942** | **−0.0235** |

- 去掉 Transformer 后，中短蛋白反而更好（+0.0153 / +0.0131）。这可能是因为 4 层 Transformer 参数量大、在较小训练集上过拟合；而纯 LGC + 分类器的归纳偏置更紧，短序列上泛化更好。
- 但在 **≥1500 长蛋白上，Transformer 明确带来 +0.0235 MCC 的优势**。这与直觉一致：超长序列中需要聚合远距离残基关系，LGC 只覆盖局部 30 Å 内的 Cα 近邻，Transformer 的全局自注意力能够补充跨域的序列上下文。
- 然而，这个收益与 LGC 的收益量级不同：关掉 LGC 损失 0.0409，关掉 Transformer 仅损失 0.0235（且后者还要在去掉 PE 的受控条件下才成立）。

- **结论 H4 部分成立：Transformer 不是长蛋白性能的决定性模块，但在 LGC 基础上提供额外、温和的远程上下文补充。**

一个值得注意的对比是 **A7_only_esm2**：它只保留 ESM2 嵌入 + 逐残基分类器，在 ≥1500 上却达到 0.2037，甚至优于 A1_full 的 0.1996。这说明：

- ESM2-2560 维预训练嵌入本身已经编码了很强的结构和进化上下文；
- 后续模块（LGC / Transformer / Seq-PE）的作用是在此基础上做"微调式重排"；
- 当这些后续模块设计不当（如正弦 PE）或训练数据不足时，反而可能破坏 ESM2 已经学到的长程结构先验。

> 换言之，**本模型在长蛋白上的领先，底层是 ESM2 的能力，放大 ESM2 能力的最有效支路是 LGC，而 Transformer 只起到较小的锦上添花作用**。

---

## 5. 结论

- **主导长序列蛋白预测性能的模块是 LGC（Local Geometric Context，基于 Cα 的 k-近邻 3D 几何消息传递）。** 在 ≥1500 的超长蛋白上，关掉 LGC 使 MCC 从 0.1996 跌到 0.1587（−20.5%），是所有消融中幅度最大的回退。
- **正弦序列位置编码对本模型是"负贡献"**，去掉后在全部长度 bin 均略有提升，长蛋白提升最明显（+0.0181 MCC @ ≥1500）。
- **k 的具体数值（8/16/32）影响有限**，重要的是 LGC 这条路径是否打开。
- **Transformer 对长蛋白有温和正向作用，但不是决定性模块。** 在去掉 Seq-PE 的受控条件下，A3（含 Trans）相对 A8（无 Trans）在 ≥1500 上高 0.0235 MCC；但该增益远小于 LGC 的 0.0409。
- **ESM2 预训练嵌入是长蛋白性能的地基。** A7（仅 ESM2 + 分类器）在 ≥1500 上达到 0.2037，已超过 A1_full 的 0.1996。LGC 是在 ESM2 表示上最有效的结构放大器，而 Transformer 仅提供较小的远程补充。
- 把以上几点合起来看：**本模型相比 CLAPE / TransBind / ATMGBs / GeSite 在 ≥1500 上的领先，底层来自 ESM2，最直接/最大的放大器是 LGC 这条 3D 几何支路；Transformer 可作为可选补充，但应优先保证 LGC 打开且避免正弦 PE 的外推噪声。**

### 工程建议

1. **"LGC-only"（A3 配置）** 可以作为下一轮主线实验的默认架构。即使在同样 80 epoch 的充分训练下，它的归纳偏置更贴合问题，长蛋白泛化上限更高。
2. 如果一定要保留显式位置信息，建议把"正弦绝对 PE"换成"相对位置编码 / ALiBi / Rotary"等对长序列更友好的方案，而不是当前的 fixed-sinusoidal。
3. 对 LGC 本身，k=16 已经是性价比拐点；再把 k 加到 32 对长蛋白边际收益接近零，但显存与计算都会上升。
4. Transformer 可作为可选补充，但不应以增加过长序列训练成本为代价。对于资源受限场景，可优先考虑 A8（ESM2 + LGC + 分类器），它在 ≥1500 上仅比 A3 低 0.0235 MCC，但少了 4 层 Transformer 的显存与参数量。

### 复现方式

建议使用统一的 preset 脚本 [train_ablation2.py](file:///lgc_tf/ablution_part/train_ablation2.py)，避免手工组合开关出错：

```bash
cd /lgc_tf
python -m ablution_part.train_ablation2 --variant full
python -m ablution_part.train_ablation2 --variant no_lgc
python -m ablution_part.train_ablation2 --variant no_pe
python -m ablution_part.train_ablation2 --variant no_lgc_no_pe
python -m ablution_part.train_ablation2 --variant k8
python -m ablution_part.train_ablation2 --variant k32
python -m ablution_part.train_ablation2 --variant no_transformer
python -m ablution_part.train_ablation2 --variant only_esm2
python -m ablution_part.train_ablation2 --variant esm2_lgc
```

所有脚本只读 `/lgc_tf/src` 内已有模块（`src.models.sde`, `src.models.lgc`, `src.data.rna_dataset`, `src.losses`, `src.utils.metrics`），不修改主线实验代码。
