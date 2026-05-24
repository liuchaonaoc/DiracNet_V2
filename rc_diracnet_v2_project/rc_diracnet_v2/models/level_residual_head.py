"""Bounded optional residual calibration head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..constants import HARTREE_eV


class LevelResidualHead(nn.Module):
    def __init__(self, d_cond: int, delta_max_meV: float = 5.0) -> None:
        super().__init__()
        self.delta_max = float(delta_max_meV) / 1000.0 / HARTREE_eV
        hidden = max(16, d_cond // 4)
        self.mlp = nn.Sequential(nn.Linear(d_cond, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h_cond: Tensor) -> Tensor:
        return self.delta_max * torch.tanh(self.mlp(h_cond).squeeze(-1))
