"""Fixed B-spline radial basis with first and second derivatives."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class BSplineBasis(nn.Module):
    def __init__(
        self,
        n_basis: int = 32,
        order: int = 3,
        r_min: float = 1.0e-4,
        r_max: float = 50.0,
        knot_scheme: str = "log",
        boundary_clamp: bool = True,
    ) -> None:
        super().__init__()
        if n_basis < order + 1:
            raise ValueError("n_basis must be at least order + 1")
        if r_min <= 0 or r_max <= r_min:
            raise ValueError("require 0 < r_min < r_max")
        self.n_basis = int(n_basis)
        self.order = int(order)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.knot_scheme = str(knot_scheme)
        self.boundary_clamp = bool(boundary_clamp)
        self.register_buffer("knots", torch.empty(0))
        self.register_buffer("B", torch.empty(0))
        self.register_buffer("dB", torch.empty(0))
        self.register_buffer("d2B", torch.empty(0))
        self._ready = False

    def _internal_knots(self) -> np.ndarray:
        n_internal = self.n_basis - self.order - 1
        if n_internal <= 0:
            return np.empty(0, dtype=np.float64)
        if self.knot_scheme == "linear":
            pts = np.linspace(self.r_min, self.r_max, n_internal + 2, dtype=np.float64)[1:-1]
        elif self.knot_scheme in ("log", "physics"):
            pts = np.geomspace(self.r_min, self.r_max, n_internal + 2, dtype=np.float64)[1:-1]
        elif self.knot_scheme == "loglinear":
            x = np.linspace(0.0, 1.0, n_internal + 2, dtype=np.float64)[1:-1]
            log_pts = np.exp(np.log(self.r_min) + (np.log(self.r_max) - np.log(self.r_min)) * x)
            lin_pts = self.r_min + (self.r_max - self.r_min) * x
            pts = (1.0 - x) * log_pts + x * lin_pts
        else:
            raise ValueError(f"unknown knot_scheme: {self.knot_scheme}")
        return pts

    def _build_knots_np(self) -> np.ndarray:
        p = self.order
        left = np.full(p + 1, self.r_min, dtype=np.float64)
        right = np.full(p + 1, self.r_max, dtype=np.float64)
        return np.concatenate([left, self._internal_knots(), right])

    @torch.no_grad()
    def precompute(self, r_grid: Tensor) -> None:
        try:
            from scipy.interpolate import BSpline
        except Exception as exc:  # pragma: no cover
            raise ImportError("scipy is required for BSplineBasis.precompute") from exc
        if r_grid.dim() != 1:
            raise ValueError("r_grid must be 1D")
        r_np = r_grid.detach().cpu().double().numpy()
        knots = self._build_knots_np()
        K, p = self.n_basis, self.order
        B = np.zeros((K, r_np.shape[0]), dtype=np.float64)
        dB = np.zeros_like(B)
        d2B = np.zeros_like(B)
        for k in range(K):
            coeff = np.zeros(K, dtype=np.float64)
            coeff[k] = 1.0
            spl = BSpline(knots, coeff, p, extrapolate=False)
            B[k] = np.nan_to_num(spl(r_np, nu=0), nan=0.0)
            dB[k] = np.nan_to_num(spl(r_np, nu=1), nan=0.0)
            d2B[k] = np.nan_to_num(spl(r_np, nu=2), nan=0.0)
        # Make the right endpoint inclusive for the last basis function.
        if abs(r_np[-1] - self.r_max) < 1e-5:
            B[:, -1] = 0.0
            B[-1, -1] = 1.0
            dB[:, -1] = 0.0
            d2B[:, -1] = 0.0
        self.knots = torch.tensor(knots, dtype=torch.float32, device=r_grid.device)
        self.B = torch.tensor(B, dtype=torch.float32, device=r_grid.device)
        self.dB = torch.tensor(dB, dtype=torch.float32, device=r_grid.device)
        self.d2B = torch.tensor(d2B, dtype=torch.float32, device=r_grid.device)
        self._ready = True

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self._ready:
            raise RuntimeError("BSplineBasis.precompute(r_grid) must be called first")
        return self.B, self.dB, self.d2B
