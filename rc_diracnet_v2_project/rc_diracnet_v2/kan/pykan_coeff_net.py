"""Optional pykan-backed coefficient network.

The external ``pykan`` package is imported lazily because V2 should remain
installable without this optional backend. The PyPI package is typically
installed as ``pykan`` but imported as ``kan``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PyKANCoeffNet(nn.Module):
    """Adapter from pykan's KAN module to the V2 coefficient-net API."""

    def __init__(
        self,
        d_cond: int = 256,
        d_phys: int = 8,
        n_basis: int = 32,
        hidden: int = 128,
        grid_size: int = 9,
        spline_order: int = 3,
        max_z: int = 110,
        max_n: int = 30,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            from kan import KAN  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "kan.kind='pykan' requires the optional pykan package. "
                "Install it with `python -m pip install pykan`, or use kan.kind='kan'."
            ) from exc

        self.d_cond = int(d_cond)
        self.d_phys = int(d_phys)
        self.n_basis = int(n_basis)
        self.max_z = float(max_z)
        self.max_n = float(max_n)

        d_in = self.d_cond + self.d_phys
        self.trunk = self._build_pykan(
            KAN,
            width=[d_in, int(hidden), int(hidden)],
            grid=int(grid_size),
            k=int(spline_order),
            seed=int(seed),
        )
        self.head_lam = nn.Linear(int(hidden), 1)
        self.head_c = nn.Linear(int(hidden), self.n_basis)
        nn.init.zeros_(self.head_lam.weight)
        nn.init.zeros_(self.head_lam.bias)
        nn.init.zeros_(self.head_c.weight)
        nn.init.zeros_(self.head_c.bias)

    def _build_pykan(self, kan_cls, **kwargs):
        """Handle minor pykan constructor differences across versions."""
        try:
            return kan_cls(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            return kan_cls(**kwargs)

    def _normalize_phys(self, x: Tensor) -> Tensor:
        scale = torch.tensor([self.max_z, 30.0, self.max_n, self.max_n, 7.0, 10.0, 14.0, 1.0], device=x.device, dtype=x.dtype)
        y = x / scale.clamp_min(1.0)
        return (2.0 * y.clamp(0.0, 1.0)) - 1.0

    def forward(self, h_cond: Tensor, per_orb_features: Tensor, orb_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch_size, n_orb, _ = per_orb_features.shape
        h = torch.tanh(h_cond).unsqueeze(1).expand(batch_size, n_orb, -1)
        phys = self._normalize_phys(per_orb_features)
        x = torch.cat([h, phys], dim=-1).reshape(batch_size * n_orb, -1)
        y = self.trunk(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        lam = self.head_lam(y).view(batch_size, n_orb)
        c = self.head_c(y).view(batch_size, n_orb, self.n_basis)
        if orb_mask is not None:
            mask = orb_mask.to(c.device).to(c.dtype)
            lam = lam * mask
            c = c * mask.unsqueeze(-1)
        return lam, c
