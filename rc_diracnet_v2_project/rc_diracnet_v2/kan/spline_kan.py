"""Small self-contained KAN layers based on learnable spline edge functions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SplineKANLayer(nn.Module):
    """Kolmogorov-Arnold layer with one learnable spline per input-output edge.

    For input ``x[..., in]`` the layer computes
    ``y[..., out] = sum_in phi[out, in](x[..., in]) + bias[out]``.
    Each ``phi`` is represented by learnable coefficients on fixed linear
    B-spline (hat) basis functions over ``[-grid_range, grid_range]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 9,
        grid_range: float = 1.0,
        init_scale: float = 1.0e-2,
        use_base: bool = True,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.grid_range = float(grid_range)
        self.use_base = bool(use_base)
        knots = torch.linspace(-self.grid_range, self.grid_range, self.grid_size)
        self.register_buffer("knots", knots)
        self.spline_weight = nn.Parameter(torch.empty(self.out_features, self.in_features, self.grid_size))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        if self.use_base:
            self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        else:
            self.register_parameter("base_weight", None)
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float = 1.0e-2) -> None:
        nn.init.normal_(self.spline_weight, mean=0.0, std=float(init_scale))
        nn.init.zeros_(self.bias)
        if self.base_weight is not None:
            nn.init.xavier_uniform_(self.base_weight)
            self.base_weight.data.mul_(float(init_scale))

    def basis(self, x: Tensor) -> Tensor:
        """Return hat-spline basis values with shape ``[..., in_features, grid]``."""
        x = x.clamp(-self.grid_range, self.grid_range)
        spacing = (2.0 * self.grid_range) / (self.grid_size - 1)
        dist = (x.unsqueeze(-1) - self.knots.to(x.device).to(x.dtype)).abs() / spacing
        return (1.0 - dist).clamp_min(0.0)

    def forward(self, x: Tensor) -> Tensor:
        basis = self.basis(x)
        y = torch.einsum("...ig,oig->...o", basis, self.spline_weight)
        if self.base_weight is not None:
            y = y + torch.einsum("...i,oi->...o", torch.nn.functional.silu(x), self.base_weight)
        return y + self.bias


class SplineKAN(nn.Module):
    """Two-layer KAN used as the V2 coefficient hypernetwork trunk."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        grid_size: int = 9,
        grid_range: float = 1.0,
        final_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer1 = SplineKANLayer(in_features, hidden_features, grid_size=grid_size, grid_range=grid_range, init_scale=0.05)
        self.layer2 = SplineKANLayer(
            hidden_features,
            out_features,
            grid_size=grid_size,
            grid_range=grid_range,
            init_scale=final_init_scale,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = torch.tanh(self.layer1(x))
        return self.layer2(h)
