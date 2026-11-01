"""双向 Mamba 模块。

数学定义：
    H_f = Mamba_f(X)
    H_b = Reverse(Mamba_b(Reverse(X)))
    H   = W_o [H_f || H_b] + b_o
    X'  = X + H
    X_out = X' + FFN(LN(X'))

传入本模块的序列必须已经按真实长度切片，不能包含 padding。
"""

from typing import Tuple

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception as error:  # 安装缺失或二进制环境不兼容时给出清晰错误。
    Mamba = None
    _MAMBA_IMPORT_ERROR = error
else:
    _MAMBA_IMPORT_ERROR = None

try:
    from mamba_ssm import Mamba2
except Exception:
    Mamba2 = None


def reverse_sequence(x: torch.Tensor) -> torch.Tensor:
    """沿序列维反转完整有效序列。

    输入/输出：FloatTensor[B,L,D]；形状与 dtype 均不改变。
    """

    if x.ndim != 3:
        raise ValueError(f"reverse_sequence 需要 [B,L,D]，实际 {tuple(x.shape)}")
    return torch.flip(x, dims=(1,))


def build_mamba(
    backend: str,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> nn.Module:
    """按照配置创建一个单向 Mamba 层，输入输出均为 [B,L,D]。"""

    if backend == "mamba":
        if Mamba is None:
            raise ImportError(
                "无法导入 mamba_ssm.Mamba。请按 requirements_bimamba.txt 配置 CUDA 环境。"
            ) from _MAMBA_IMPORT_ERROR
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend == "mamba2":
        if Mamba2 is None:
            raise ImportError("当前 mamba_ssm 未提供 Mamba2，请改用 mamba 或升级依赖。")
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    raise ValueError(f"未知 Mamba 后端：{backend!r}，应为 'mamba' 或 'mamba2'")


class BiMambaBlock(nn.Module):
    """Concat+Linear 融合的双向 Mamba 残差块。

    输入：x[B,L,d_model]，其中 L 范围内全部为真实残基。
    输出：与输入同形状的 x_out[B,L,d_model]。
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        backend: str = "mamba",
    ):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.forward_mamba = build_mamba(backend, d_model, d_state, d_conv, expand)
        self.backward_mamba = build_mamba(backend, d_model, d_state, d_conv, expand)
        self.direction_fusion = nn.Linear(2 * d_model, d_model)
        self.direction_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def directional_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算正向和反向状态。

        输入：无 padding 的 x[B,L,D]。
        输出：(h_forward, h_backward)，二者形状均为 [B,L,D]，并已对齐原序列顺序。
        """

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"BiMamba 输入应为 [B,L,{self.d_model}]，实际 {tuple(x.shape)}"
            )
        z = self.norm1(x)
        h_forward = self.forward_mamba(z)
        h_backward = reverse_sequence(self.backward_mamba(reverse_sequence(z)))
        assert h_forward.shape == h_backward.shape == x.shape
        assert h_forward.shape[-1] == self.d_model
        return h_forward, h_backward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_forward, h_backward = self.directional_states(x)
        h = self.direction_fusion(torch.cat((h_forward, h_backward), dim=-1))
        x_residual = x + self.direction_dropout(h)
        return x_residual + self.ffn(self.norm2(x_residual))

