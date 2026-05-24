"""Analytic origin/asymptotic envelope for V2."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..constants import ALPHA


class AnalyticEnvelope(nn.Module):
    def __init__(self, alpha: float = ALPHA, use_relativistic_gamma: bool = False) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.use_relativistic_gamma = bool(use_relativistic_gamma)

    def _gamma(self, kappa: Tensor, Z: Tensor, dtype: torch.dtype) -> Tensor:
        kap = kappa.to(dtype).abs().clamp_min(1.0)
        if not self.use_relativistic_gamma:
            return kap
        za = Z.to(dtype).view(-1, 1) * self.alpha
        return (kap * kap - za * za).clamp_min(1.0e-6).sqrt()

    def forward(self, r_grid: Tensor, lam: Tensor, kappa: Tensor, Z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r = r_grid.to(lam.device).to(lam.dtype)
        r_b = r.view(1, 1, -1)
        lam_b = lam.to(lam.dtype).unsqueeze(-1)
        gamma = self._gamma(kappa.to(lam.device), Z.to(lam.device), lam.dtype).unsqueeze(-1)
        log_env = gamma * torch.log(r_b) - lam_b * r_b
        env = torch.exp(log_env)
        d_log = gamma / r_b - lam_b
        denv = d_log * env
        d2_log = -gamma / (r_b * r_b)
        d2env = (d2_log + d_log * d_log) * env
        return env, denv, d2env
