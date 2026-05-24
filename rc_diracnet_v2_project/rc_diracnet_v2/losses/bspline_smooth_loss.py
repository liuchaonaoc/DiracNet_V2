"""Smoothness loss on B-spline coefficients."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BSplineSmoothLoss(nn.Module):
    def forward(self, c: Tensor, orb_mask: Tensor) -> Tensor:
        if c.shape[-1] < 3:
            return c.sum() * 0.0
        d2 = c[..., 2:] - 2.0 * c[..., 1:-1] + c[..., :-2]
        sq = d2.pow(2).sum(dim=-1)
        mask = orb_mask.to(c.device).to(sq.dtype)
        return (sq * mask).sum() / mask.sum().clamp_min(1.0)
