"""Multi-task loss balancer (manual / GradNorm)."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


class LossBalancer(nn.Module):
    """加权多任务损失。

    支持策略：
        * ``manual``  : 静态权重；
        * ``gradnorm``: 基于损失下降率自动调整（最小实现）。
    """

    def __init__(
        self,
        strategy: str = "manual",
        init_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if init_weights is None:
            init_weights = {}
        self.strategy = strategy
        self.keys = sorted(init_weights.keys())
        log_w = torch.log(torch.tensor([max(init_weights[k], 1e-6) for k in self.keys]))
        if strategy == "gradnorm":
            self.log_w = nn.Parameter(log_w)
        else:
            self.register_buffer("log_w", log_w)

    def combine(self, losses: dict[str, Tensor]) -> Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for k, v in losses.items():
            if k not in self.keys:
                continue
            idx = self.keys.index(k)
            w = torch.exp(self.log_w[idx])
            total = total + w * v
        return total

    def detached_weights(self) -> dict[str, float]:
        return {k: float(torch.exp(self.log_w[i])) for i, k in enumerate(self.keys)}
