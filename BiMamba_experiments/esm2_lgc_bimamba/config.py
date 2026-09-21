"""实验配置。

直接修改本文件中的路径和参数，然后在 IDE 中运行 ``train_bimamba.py`` 或
``infer_bimamba.py``。两个入口均不读取命令行参数。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BRANCH_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    """数据路径配置；每个目录应直接包含单蛋白 ``.pt`` 文件。"""

    train_dir: Path = Path("/root/autodl-tmp/features/merged/RNA-495_Train")
    # 设为 None 时，按 val_ratio 从训练集固定切分验证集。
    val_dir: Optional[Path] = None
    test_dir: Path = Path("/root/autodl-tmp/features/merged/RNA-117_Test")
    val_ratio: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    """ESM2 + LGC + BiMamba 的结构参数。"""

    esm2_dim: int = 2560
    d_model: int = 256
    num_bimamba_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    classifier_hidden_dim: int = 128
    sde_dim: int = 16
    cutoff: float = 30.0
    k_neighbors: int = 16
    # 默认使用文档指定的 mamba_ssm.Mamba；如环境只支持 Mamba2，可改为 "mamba2"。
    mamba_backend: str = "mamba"


@dataclass(frozen=True)
class TrainConfig:
    """训练参数；保持与当前 Transformer 基线一致。"""

    seed: int = 42
    epochs: int = 80
    patience: int = 10
    batch_size: int = 2
    accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    num_workers: int = 2
    use_amp: bool = True
    focal_gamma: float = 2.0
    threshold_start: float = 0.01
    threshold_end: float = 0.99
    threshold_step: float = 0.01


@dataclass(frozen=True)
class InferenceConfig:
    """测试集推理配置。运行时间统计要求 batch_size 固定为 1。"""

    num_workers: int = 0
    use_amp: bool = True
    # 可选：填入原 Transformer 基线 checkpoint 后，会同时输出基线对比结果。
    baseline_checkpoint: Optional[Path] = None


@dataclass(frozen=True)
class OutputConfig:
    """本分支的全部输出目录。"""

    checkpoint_dir: Path = BRANCH_DIR / "checkpoints"
    log_dir: Path = BRANCH_DIR / "logs"
    result_dir: Path = BRANCH_DIR / "results" / "results_495-1346"
    best_checkpoint_name: str = "esm2_lgc_bimamba_best_newloss.pt"


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
INFERENCE = InferenceConfig()
OUTPUT = OutputConfig()

