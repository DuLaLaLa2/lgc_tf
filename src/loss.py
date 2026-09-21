import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """
    原始二分类 Focal Loss。

    该类保留用于原实验复现和后续消融实验，
    不建议删除或修改原有行为。

    参数
    ----------
    alpha : float
        正类 alpha 权重。
        正样本权重为 alpha，
        负样本权重为 1 - alpha。

    gamma : float
        Focal Loss 聚焦参数。
        gamma 越大，对容易分类样本的抑制越强。

    reduction : str
        损失聚合方式：
        - "mean"：求平均
        - "sum"：求和
        - "none"：返回逐样本损失
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha 必须位于 [0, 1]，当前值为 {alpha}"
            )

        if gamma < 0.0:
            raise ValueError(
                f"gamma 必须 >= 0，当前值为 {gamma}"
            )

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"不支持的 reduction={reduction}"
            )

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算原始 Binary Focal Loss。

        参数
        ----------
        logits : torch.Tensor
            模型输出的原始 logits。
            支持形状 [N]、[N, 1] 等。

        targets : torch.Tensor
            二分类真实标签，取值为 0 或 1。
            形状应与 logits 一致。

        返回
        ----------
        torch.Tensor
            根据 reduction 返回标量或逐样本 loss。
        """

        targets = targets.float()

        # 使用 with_logits 版本保证数值稳定性，
        # 不需要提前对 logits 做 sigmoid。
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)

        # pt 表示模型对“真实类别”的预测概率：
        # y=1 时 pt=p；
        # y=0 时 pt=1-p。
        pt = (
            probs * targets
            + (1.0 - probs) * (1.0 - targets)
        )

        # 正类使用 alpha，
        # 负类使用 1-alpha。
        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        # Focal modulation：
        # 容易样本 pt≈1，因此权重趋近于 0；
        # 困难样本 pt 较低，因此保留更大的梯度。
        focal_weight = (1.0 - pt).pow(self.gamma)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class GlobalWeightedFocalLossWithLogits(nn.Module):
    """
    面向残基级 RNA 结合位点预测的
    “全局类别权重 + Focal 动态难样本权重”损失函数。

    设计目的
    ----------
    当前任务存在严重类别不平衡，例如：

        RNA-binding residue : non-binding residue
        ≈ 1 : 15

    本损失函数把“不平衡处理”拆成两个相互独立的部分：

    1. 全局类别权重：
       提高少数类 RNA-binding residues 的损失贡献。

    2. Focal 动态权重：
       抑制大量已经被正确分类的 easy negatives，
       将训练梯度进一步集中到：
       - 难正样本
       - 假阴性趋势样本
       - 难负样本

    最终形式：

        Loss_i =
            class_weight_i
            × (1 - pt_i)^gamma
            × BCE_i

    其中：

        positive class weight = pos_weight
        negative class weight = 1.0

    与传统 Focal Loss 的区别
    ----------
    本实现不再额外使用 alpha。

    原因是 pos_weight 已经负责类别不平衡；
    如果再次引入 alpha，会形成重复的类别补偿，
    容易让少数类权重过大。

    参数
    ----------
    pos_weight : float
        正类的全局权重。

        推荐按照训练集统计：

            pos_weight =
                min(
                    sqrt(N_neg / N_pos),
                    max_pos_weight
                )

        当正负比例约为 1:15：

            sqrt(15) ≈ 3.873

    gamma : float
        Focal 聚焦参数。

        推荐第一版：
            gamma = 2.0

    reduction : str
        损失聚合方法：
        - "mean"
        - "sum"
        - "none"

    eps : float
        数值稳定保护值。
    """

    def __init__(
        self,
        pos_weight: float,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()

        if pos_weight <= 0.0:
            raise ValueError(
                "pos_weight 必须大于 0，"
                f"当前值为 {pos_weight}"
            )

        if gamma < 0.0:
            raise ValueError(
                "gamma 必须大于等于 0，"
                f"当前值为 {gamma}"
            )

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"不支持的 reduction={reduction}，"
                "只能是 mean、sum 或 none。"
            )

        self.pos_weight = float(pos_weight)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Global Weighted Focal Loss。

        参数
        ----------
        logits : torch.Tensor
            模型输出的 logits。

            在当前项目中的实际输入通常为：

                logits[mask]

            因此一般是一维有效残基序列：

                [N_valid_residues]

        targets : torch.Tensor
            与 logits 对应的二分类标签：

                0 = 非 RNA 结合残基
                1 = RNA 结合残基

        返回
        ----------
        torch.Tensor
            reduction="mean" 时返回一个标量 loss，
            可直接执行：

                loss.backward()

            完全兼容当前训练代码：

                loss = criterion(
                    logits[mask],
                    y[mask]
                )
        """

        # -----------------------------------------------------
        # 1. 基础输入检查
        # -----------------------------------------------------

        if logits.shape != targets.shape:
            raise ValueError(
                "logits 与 targets 的形状必须完全一致，"
                f"当前 logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        # 标签转换为与 logits 一致的数据类型。
        # 在 AMP 场景下可保持与模型输出兼容。
        targets = targets.to(dtype=logits.dtype)

        # -----------------------------------------------------
        # 2. 计算逐残基 BCE
        # -----------------------------------------------------
        #
        # 必须使用 binary_cross_entropy_with_logits，
        # 而不是先 sigmoid 再 BCE。
        #
        # 这样在概率接近 0 或 1 时数值更加稳定。
        # -----------------------------------------------------

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # -----------------------------------------------------
        # 3. 计算模型概率
        # -----------------------------------------------------

        probs = torch.sigmoid(logits)

        # -----------------------------------------------------
        # 4. 计算 pt
        # -----------------------------------------------------
        #
        # pt 表示模型对真实类别的预测概率。
        #
        # 如果 y=1：
        #     pt = p
        #
        # 如果 y=0：
        #     pt = 1-p
        #
        # 因此：
        #
        # easy sample:
        #     pt → 1
        #
        # hard sample:
        #     pt → 0
        # -----------------------------------------------------

        pt = (
            targets * probs
            + (1.0 - targets) * (1.0 - probs)
        )

        # 为了防止极端浮点数造成潜在数值问题，
        # 将 pt 限制在有效范围。
        pt = pt.clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )

        # -----------------------------------------------------
        # 5. Focal 动态权重
        # -----------------------------------------------------
        #
        # gamma=2 时：
        #
        # pt = 0.95
        # weight = 0.05² = 0.0025
        #
        # pt = 0.50
        # weight = 0.50² = 0.25
        #
        # pt = 0.20
        # weight = 0.80² = 0.64
        #
        # 因此大量 easy negatives 会被显著抑制，
        # 困难残基保留较大的梯度。
        # -----------------------------------------------------

        focal_weight = (1.0 - pt).pow(self.gamma)

        # -----------------------------------------------------
        # 6. 全局类别权重
        # -----------------------------------------------------
        #
        # 正类：
        #     pos_weight
        #
        # 负类：
        #     1.0
        #
        # 例如当前数据约为：
        #
        #     positive : negative ≈ 1 : 15
        #
        # 推荐：
        #
        #     pos_weight = sqrt(15)
        #                ≈ 3.873
        #
        # 而不是直接使用 15，
        # 避免过度补偿导致：
        #
        #     recall 大涨
        #     precision 崩溃
        # -----------------------------------------------------

        positive_weight = torch.as_tensor(
            self.pos_weight,
            dtype=logits.dtype,
            device=logits.device,
        )

        class_weight = torch.where(
            targets > 0.5,
            positive_weight,
            torch.ones_like(targets),
        )

        # -----------------------------------------------------
        # 7. 最终 Minority-Aware Loss
        # -----------------------------------------------------

        loss = (
            class_weight
            * focal_weight
            * bce
        )

        # -----------------------------------------------------
        # 8. reduction
        # -----------------------------------------------------

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def compute_sqrt_pos_weight(
    positive_count: float,
    negative_count: float,
    max_pos_weight: float = 5.0,
    min_pos_weight: float = 1.0,
) -> float:
    """
    根据训练集正负残基数量计算温和的全局正类权重。

    设计公式
    ----------
        ratio =
            N_negative / N_positive

        pos_weight =
            sqrt(ratio)

    并限制到：

        [min_pos_weight, max_pos_weight]

    为什么不用 N_negative / N_positive？
    -------------------------------------
    如果当前正负比例约为：

        1 : 15

    直接使用：

        pos_weight = 15

    会使正类损失权重过强。

    对残基级结合位点预测而言，
    很容易产生：

        Recall ↑↑
        Precision ↓↓
        False Positive ↑↑

    因此第一阶段实验采用平方根压缩：

        sqrt(15) ≈ 3.873

    在增强少数类关注的同时，
    减少过度补偿风险。

    参数
    ----------
    positive_count : float
        训练集正类残基数量。

    negative_count : float
        训练集负类残基数量。

    max_pos_weight : float
        正类权重上限。
        第一版推荐 5.0。

    min_pos_weight : float
        正类权重下限。
        默认 1.0。

    返回
    ----------
    float
        最终正类全局权重。
    """

    positive_count = float(positive_count)
    negative_count = float(negative_count)

    if positive_count <= 0:
        raise ValueError(
            "positive_count 必须大于 0，"
            f"当前值为 {positive_count}"
        )

    if negative_count <= 0:
        raise ValueError(
            "negative_count 必须大于 0，"
            f"当前值为 {negative_count}"
        )

    if min_pos_weight <= 0:
        raise ValueError(
            "min_pos_weight 必须大于 0。"
        )

    if max_pos_weight < min_pos_weight:
        raise ValueError(
            "max_pos_weight 不能小于 min_pos_weight。"
        )

    imbalance_ratio = (
        negative_count / positive_count
    )

    raw_pos_weight = math.sqrt(
        imbalance_ratio
    )

    pos_weight = max(
        min_pos_weight,
        min(raw_pos_weight, max_pos_weight),
    )

    return float(pos_weight)