"""KAN coefficient network for V2.

This module intentionally uses a real KAN-style parameterisation: every layer is
a sum of learnable univariate spline edge functions, not an MLP fallback.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .spline_kan import SplineKAN


class KANCoeffNet(nn.Module):
    def __init__(
        self,
        d_cond: int = 256,
        d_phys: int = 8,
        n_basis: int = 32,
        hidden: int = 128,
        grid_size: int = 9,
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
        self.net = SplineKAN(
            in_features=self.d_cond + self.d_phys,
            hidden_features=int(hidden),
            out_features=1 + self.n_basis,
            grid_size=int(grid_size),
            final_init_scale=0.0,
        )

    def _normalize_phys(self, x: Tensor) -> Tensor:
        scale = torch.tensor([self.max_z, 30.0, self.max_n, self.max_n, 7.0, 10.0, 14.0, 1.0], device=x.device, dtype=x.dtype)
        y = x / scale.clamp_min(1.0)
        return (2.0 * y.clamp(0.0, 1.0)) - 1.0

    def forward(self, h_cond: Tensor, per_orb_features: Tensor, orb_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch_size, n_orb, _ = per_orb_features.shape
        # Keep KAN inputs in the spline grid range. The physics features are
        # explicitly scaled; h_cond is softly bounded rather than clipped.
        h = torch.tanh(h_cond).unsqueeze(1).expand(batch_size, n_orb, -1)
        phys = self._normalize_phys(per_orb_features)
        x = torch.cat([h, phys], dim=-1).reshape(batch_size * n_orb, -1)
        y = self.net(x).view(batch_size, n_orb, 1 + self.n_basis)
        lam = y[..., 0]
        c = y[..., 1:]
        if orb_mask is not None:
            mask = orb_mask.to(c.device).to(c.dtype)
            lam = lam * mask
            c = c * mask.unsqueeze(-1)
        return lam, c
