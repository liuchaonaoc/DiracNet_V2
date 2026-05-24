"""Slater Local Density Approximation (LDA).

对应 frame_RC_V1_1.md §4.8 slater_lda.py。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def compute_density(
    P: Tensor, Q: Tensor, occ: Tensor, grid,
) -> Tensor:
    """球对称总电荷密度 ρ(r) = Σ_a occ_a * (P_a²+Q_a²) / (4π r²)。

    P, Q : [B, N_orb, N_grid]
    occ  : [B, N_orb]
    """
    r2 = grid.r.to(P.dtype).to(P.device) ** 2 * 4.0 * math.pi
    rho_radial = (P * P + Q * Q) * occ.unsqueeze(-1)
    rho = rho_radial.sum(dim=1) / r2.view(1, -1)
    return rho


def slater_exchange_potential(rho: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Slater LDA 交换势 V_x = -(3/π)^{1/3} · ρ^{1/3}.

    Parameters
    ----------
    rho : Tensor [B, N_grid]   非负电荷密度。
    eps : float
        backward 稳定性下限：``ρ.clamp_min(eps)`` 避免
        ``dV/dρ = -C/3 · ρ^{-2/3}`` 在 ρ=0 处奇异（INR/RC 输出在外层
        网格上常常精确下溢到 0）。

    Returns
    -------
    Tensor [B, N_grid]
    """
    c = -((3.0 / math.pi) ** (1.0 / 3.0))
    return c * rho.clamp_min(eps) ** (1.0 / 3.0)
