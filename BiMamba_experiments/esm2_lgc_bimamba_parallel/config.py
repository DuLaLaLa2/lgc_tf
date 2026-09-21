"""并行分支配置；修改路径后直接运行训练或推理文件。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BRANCH_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    train_dir: Path = Path("/root/autodl-tmp/features/merged/RNA-495_Train")
    # None 表示按固定 seed 从训练目录划分 20% 验证集。
    val_dir: Optional[Path] = None
    test_dir: Path = Path("/root/autodl-tmp/features/merged/RNA_1346_New")
    val_ratio: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
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
    mamba_backend: str = "mamba"


@dataclass(frozen=True)
class TrainConfig:
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
    num_workers: int = 0
    use_amp: bool = True
    # 可选：填写 Transformer 基线 checkpoint 后输出同测试集对比。
    baseline_checkpoint: Optional[Path] = None


@dataclass(frozen=True)
class OutputConfig:
    checkpoint_dir: Path = BRANCH_DIR / "checkpoints"
    log_dir: Path = BRANCH_DIR / "logs"
    result_dir: Path = BRANCH_DIR / "results"
    best_checkpoint_name: str = "esm2_lgc_bimamba_parallel_best_newloss.pt"


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
INFERENCE = InferenceConfig()
OUTPUT = OutputConfig()

