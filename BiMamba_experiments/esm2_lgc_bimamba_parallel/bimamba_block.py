"""无 padding 输入的双向 Mamba 残差块。

H_f = Mamba_f(LN(X))
H_b = Reverse(Mamba_b(Reverse(LN(X))))
H = W_o[H_f || H_b] + b_o
X_out = X + H + FFN(...)
"""

from typing import Tuple

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception as error:
    Mamba = None
    _MAMBA_IMPORT_ERROR = error
else:
    _MAMBA_IMPORT_ERROR = None

try:
    from mamba_ssm import Mamba2    
except Exception:
    Mamba2 = None


def reverse_sequence(x: torch.Tensor) -> torch.Tensor:
    """沿序列维反转 x[B,L,D]，输出形状不变。"""

    if x.ndim != 3:
        raise ValueError(f"需要 [B,L,D]，实际 {tuple(x.shape)}")
    return torch.flip(x, dims=(1,))


def build_mamba(backend: str, d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    """创建输入输出均为 [B,L,D] 的单向 Mamba 层。"""

    if backend == "mamba":
        if Mamba is None:
            raise ImportError("无法导入 mamba_ssm.Mamba，请配置 requirements_bimamba.txt。") from _MAMBA_IMPORT_ERROR
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend == "mamba2":
        if Mamba2 is None:
            raise ImportError("当前 mamba_ssm 未提供 Mamba2。")
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    raise ValueError(f"未知 Mamba 后端：{backend}")


class BiMambaBlock(nn.Module):
    """Forward/Backward Mamba 经 Concat+Linear 融合的残差块。"""

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
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def directional_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入无 padding 的 x[B,L,D]，返回对齐原顺序的正、反向状态。"""

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"应为 [B,L,{self.d_model}]，实际 {tuple(x.shape)}")
        z = self.norm1(x)
        forward_state = self.forward_mamba(z)
        backward_state = reverse_sequence(self.backward_mamba(reverse_sequence(z)))
        assert forward_state.shape == backward_state.shape == x.shape
        return forward_state, backward_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_state, backward_state = self.directional_states(x)
        bidirectional = self.fusion(torch.cat((forward_state, backward_state), dim=-1))
        residual = x + self.dropout(bidirectional)
        return residual + self.ffn(self.norm2(residual))

