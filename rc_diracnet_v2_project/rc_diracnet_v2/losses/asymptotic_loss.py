"""Asymptotic tail loss."""

from __future__ import annotations

from torch import Tensor, nn


class AsymptoticTailLoss(nn.Module):
    def __init__(self, tail_window: int = 32) -> None:
        super().__init__()
        self.tail_window = int(tail_window)

    def forward(self, P: Tensor, r_grid: Tensor, orb_mask: Tensor) -> Tensor:
        w = min(self.tail_window, P.shape[-1])
        tail = P[..., -w:]
        peak = P.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-12)
        rel = (tail.abs() / peak).max(dim=-1).values
        mask = orb_mask.to(P.device).to(rel.dtype)
        return (rel * mask).sum() / mask.sum().clamp_min(1.0)
