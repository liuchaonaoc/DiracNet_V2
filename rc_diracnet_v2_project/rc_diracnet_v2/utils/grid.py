"""Radial grid + integration weights.

对应 frame_RC_V1_1.md §4.2 grid.py。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RadialGrid(nn.Module):
    """非均匀径向网格 + 复合梯形积分权重。

    所有量都作为 buffer 注册，`requires_grad=False`。

    Attributes
    ----------
    r   : Tensor [N_grid]     径向坐标 (a.u.)
    dr  : Tensor [N_grid]     复合梯形积分权重 (Δr_i)
    jac : Tensor [N_grid]     d r / d x，loglinear 时用
    """

    def __init__(
        self,
        r_min: float = 1.0e-5,
        r_max: float = 50.0,
        n_grid: int = 1024,
        scheme: str = "loglinear",
    ) -> None:
        super().__init__()
        if r_min <= 0.0:
            raise ValueError("r_min must be > 0 to avoid Coulomb singularity")
        if n_grid < 4:
            raise ValueError("n_grid must be >= 4")

        self.r_min = r_min
        self.r_max = r_max
        self.n_grid = n_grid
        self.scheme = scheme

        x = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
        if scheme == "linear":
            r = r_min + (r_max - r_min) * x
            jac = torch.full_like(r, (r_max - r_min))
        elif scheme == "log":
            log_min = torch.log(torch.tensor(r_min, dtype=torch.float64))
            log_max = torch.log(torch.tensor(r_max, dtype=torch.float64))
            r = torch.exp(log_min + (log_max - log_min) * x)
            jac = r * (log_max - log_min)
        elif scheme == "loglinear":
            log_min = torch.log(torch.tensor(r_min, dtype=torch.float64))
            log_max = torch.log(torch.tensor(r_max, dtype=torch.float64))
            r_log = torch.exp(log_min + (log_max - log_min) * x)
            r_lin = r_min + (r_max - r_min) * x
            mix = x  # 内层多取 log，外层多取 lin
            r = (1.0 - mix) * r_log + mix * r_lin
            dr_dx_log = r_log * (log_max - log_min)
            dr_dx_lin = torch.full_like(r_lin, r_max - r_min)
            jac = (1.0 - mix) * dr_dx_log + mix * dr_dx_lin
        else:
            raise ValueError(f"unknown scheme {scheme}")

        dr = self._composite_trapezoid_weights(r)

        self.register_buffer("r", r.to(torch.float32))
        self.register_buffer("dr", dr.to(torch.float32))
        self.register_buffer("jac", jac.to(torch.float32))

    @staticmethod
    def _composite_trapezoid_weights(r: Tensor) -> Tensor:
        """生成复合梯形权重 w_i s.t. ∑ w_i f_i ≈ ∫_{r_min}^{r_max} f(r) dr."""
        n = r.shape[0]
        w = torch.zeros_like(r)
        w[1:-1] = 0.5 * (r[2:] - r[:-2])
        w[0] = 0.5 * (r[1] - r[0])
        w[-1] = 0.5 * (r[-1] - r[-2])
        return w

    def integrate(self, f: Tensor, dim: int = -1) -> Tensor:
        """∫ f(r) dr 沿指定 dim。

        f shape : [..., N_grid]
        return  : [...]
        """
        if f.shape[dim] != self.n_grid:
            raise ValueError(
                f"integrate: f.shape[{dim}] = {f.shape[dim]} != n_grid = {self.n_grid}"
            )
        w = self.dr.to(f.dtype).to(f.device)
        return torch.sum(f * w, dim=dim)

    def cumulative_integrate(self, f: Tensor) -> Tensor:
        """∫_{r_min}^{r_i} f(r') dr'，输入输出同形状 [..., N_grid]，可微。"""
        if f.shape[-1] != self.n_grid:
            raise ValueError("cumulative_integrate: last dim must be N_grid")
        w = self.dr.to(f.dtype).to(f.device)
        return torch.cumsum(f * w, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"r_min={self.r_min}, r_max={self.r_max}, "
            f"n_grid={self.n_grid}, scheme={self.scheme}"
        )
