"""Radial Dirac operator H_D.

对应 frame_RC_V1_1.md §4.8 dirac_operator.py。

公式（原子单位制，c = 1/α）
---------------------------
对每个 (n, κ) 旋量 (P(r), Q(r))：
    H_D · (P, Q)^T = (LP, LQ)^T
    LP = V·P + c (-dQ/dr + κ/r · Q)
    LQ = (V - 2c²)·Q + c ( dP/dr + κ/r · P )

实现要求：
    * 接收 ``dPdr, dQdr`` 由 readout 解析提供（避免再做 autograd）；
    * V_eff 由 (-Z/r) + V_ee + V_xc 在外部组装；
    * κ 张量 [B, N_orb] 必须是相对论 κ（非自旋-轨道编号）。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..constants import C_LIGHT


class DiracRadialOperator:
    """无状态算子（不含可学习参数）。"""

    def __init__(self, c: float = C_LIGHT) -> None:
        self.c = c

    def apply(
        self,
        P: Tensor,                  # [B, N_orb, N_grid]
        Q: Tensor,
        dPdr: Tensor,
        dQdr: Tensor,
        V_eff: Tensor,              # [B, N_grid]
        kappa: Tensor,              # [B, N_orb]
        r_grid: Tensor,             # [N_grid]
    ) -> tuple[Tensor, Tensor]:
        """返回 (LP, LQ) = H_D · (P, Q)。"""
        c = self.c
        # κ/r 项
        inv_r = 1.0 / r_grid.view(1, 1, -1)               # [1, 1, N_grid]
        kappa_over_r = kappa.unsqueeze(-1).to(P.dtype) * inv_r  # [B, N_orb, N_grid]

        V_b = V_eff.unsqueeze(1)                          # [B, 1, N_grid]
        LP = V_b * P + c * (-dQdr + kappa_over_r * Q)
        LQ = (V_b - 2.0 * c * c) * Q + c * (dPdr + kappa_over_r * P)
        return LP, LQ


def orbital_energy_from_dirac(
    P: Tensor, Q: Tensor,
    LP: Tensor, LQ: Tensor,
    grid,
) -> Tensor:
    """通过 ⟨ψ|H_D|ψ⟩ / ⟨ψ|ψ⟩ 抽取单电子能级 E_orb [B, N_orb]。"""
    num = grid.integrate(P * LP + Q * LQ, dim=-1)
    den = grid.integrate(P * P + Q * Q, dim=-1).clamp_min(1e-12)
    return num / den
