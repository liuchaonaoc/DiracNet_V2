"""Dirac PDE residual loss.

对应 frame_RC_V1_1.md §4.10 pde_loss.py。

数值缩放说明
------------
相对论 Dirac 方程含 ``-2c²·Q`` 项 (c ≈ 137)，使得 LQ 的尺度比 LP 大 ``O(c²)``。
为防止 backward 时该项压垮其它梯度，本实现：
    * 把 ``LQ - E·Q`` 残差除以 ``2c²``（等价于对小分量做无量纲化）；
    * 大分量残差直接保留；
最终残差量级在 1 附近，对训练稳定性友好。

ADR-017 — Per-sample 相对化归一化（slice 2 入口）
-------------------------------------------------
原始 ``per_orb`` 残差（``∫ (res_P² + res_Q²) dr``）天然带能量量纲²。当 batch
里同时含 ``(Z=3, n=1)`` 与 ``(Z=1, n=4)`` 时，前者绝对残差比后者大约 ``150²``
倍，``.mean()`` 让深态梯度淹没浅态。``forward`` 接受可选的 ``e_char_sq``
（形状 ``[B, N_orb]``，特征能量² ``E_char² = (max(λ², λ_min²)/2)²``）；
若给定，per-(B, N_orb) 残差先除以 ``e_char_sq.detach()``，再做 mask-mean。
``e_char_sq=None`` 时退化为原行为，仅用于 ADR-017 前的回归对照。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..constants import C_LIGHT
from ..physics.dirac_operator import DiracRadialOperator


class DiracPDELoss(nn.Module):
    """‖ H_D ψ − E_orb ψ ‖² 归一到每个 (r, orb) 点。

    Eigenfunction enforcement
    -------------------------
    With ``detach_energy=True`` (default), ``E_orb`` is treated as a fixed
    scalar in the residual (gradient flows only through ψ). This is essential
    when ``E_orb`` is obtained from a Rayleigh quotient ``<ψ|H|ψ>/<ψ|ψ>`` of
    the same ψ: the first-order term of the residual is then automatically
    minimised by ANY ψ, robbing the PDE loss of its eigenfunction-selecting
    power. Detaching E forces the residual to favour true eigenfunctions of H
    rather than arbitrary states whose Rayleigh value happens to equal E.
    """

    def __init__(
        self,
        small_component_scale: float | None = None,
        detach_energy: bool = True,
    ) -> None:
        super().__init__()
        self.op = DiracRadialOperator()
        self.small_scale = small_component_scale or (2.0 * C_LIGHT * C_LIGHT)
        self.detach_energy = bool(detach_energy)

    def forward(
        self,
        P: Tensor, Q: Tensor,
        dPdr: Tensor, dQdr: Tensor,
        E_orb: Tensor,           # [B, N_orb]
        V_eff: Tensor,           # [B, N_grid]
        kappa: Tensor,           # [B, N_orb]
        r_grid: Tensor,          # [N_grid]
        grid,                    # RadialGrid
        orb_mask: Tensor,        # [B, N_orb]
        e_char_sq: Tensor | None = None,   # ADR-017: characteristic energy² per (B, N_orb)
    ) -> Tensor:
        LP, LQ = self.op.apply(P, Q, dPdr, dQdr, V_eff, kappa, r_grid)
        e_for_res = E_orb.detach() if self.detach_energy else E_orb
        e_exp = e_for_res.unsqueeze(-1)
        res_P = (LP - e_exp * P)
        res_Q = (LQ - e_exp * Q) / self.small_scale
        res = res_P * res_P + res_Q * res_Q
        per_orb = grid.integrate(res, dim=-1)                      # [B, N_orb]
        if e_char_sq is not None:
            per_orb = per_orb / e_char_sq.detach().clamp_min(1.0e-12)
        mask = orb_mask.float()
        denom = mask.sum().clamp_min(1.0)
        return (per_orb * mask).sum() / denom
