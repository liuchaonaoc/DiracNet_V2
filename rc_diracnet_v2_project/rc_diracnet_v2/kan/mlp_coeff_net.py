"""MLP fallback coefficient network for V2."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPCoeffNet(nn.Module):
    def __init__(
        self,
        d_cond: int = 256,
        d_phys: int = 8,
        n_basis: int = 32,
        hidden: int = 128,
        max_z: int = 110,
        max_n: int = 30,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_cond = int(d_cond)
        self.d_phys = int(d_phys)
        self.n_basis = int(n_basis)
        self.max_z = float(max_z)
        self.max_n = float(max_n)
        d_in = self.d_cond + self.d_phys
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head_lam = nn.Linear(hidden, 1)
        self.head_c = nn.Linear(hidden, self.n_basis)
        nn.init.zeros_(self.head_lam.weight); nn.init.zeros_(self.head_lam.bias)
        nn.init.zeros_(self.head_c.weight); nn.init.zeros_(self.head_c.bias)

    def _normalize_phys(self, x: Tensor) -> Tensor:
        scale = torch.tensor([self.max_z, 30.0, self.max_n, self.max_n, 7.0, 10.0, 14.0, 1.0], device=x.device, dtype=x.dtype)
        y = x / scale.clamp_min(1.0)
        return (2.0 * y.clamp(0.0, 1.0)) - 1.0

    def forward(self, h_cond: Tensor, per_orb_features: Tensor, orb_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, N, _ = per_orb_features.shape
        h = h_cond.unsqueeze(1).expand(B, N, -1)
        phys = self._normalize_phys(per_orb_features)
        x = torch.cat([h, phys], dim=-1).reshape(B * N, -1)
        y = self.trunk(x)
        lam = self.head_lam(y).view(B, N)
        c = self.head_c(y).view(B, N, self.n_basis)
        if orb_mask is not None:
            mask = orb_mask.to(c.device).to(c.dtype)
            lam = lam * mask
            c = c * mask.unsqueeze(-1)
        return lam, c
