"""Bohr-Sommerfeld radial action quantisation loss."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BohrSommerfeldActionLoss(nn.Module):
    def __init__(self, eps: float = 1.0e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, E_orb: Tensor, V_eff: Tensor, n_idx: Tensor, l_idx: Tensor, orb_mask: Tensor, grid) -> Tensor:
        r = grid.r.to(E_orb.device).to(E_orb.dtype)
        r_b = r.view(1, 1, -1)
        l = l_idx.to(E_orb.device).to(E_orb.dtype)
        centrifugal = (l * (l + 1.0)).unsqueeze(-1) / (2.0 * r_b * r_b)
        v_total = V_eff.to(E_orb.device).to(E_orb.dtype).unsqueeze(1) + centrifugal
        p_sq = 2.0 * (E_orb.unsqueeze(-1) - v_total)
        p_r = (p_sq.clamp_min(0.0) + self.eps).sqrt()
        S = grid.integrate(p_r, dim=-1)
        target = math.pi * (n_idx.to(E_orb.device).to(E_orb.dtype) - l - 0.5).clamp_min(0.0)
        err = S - target
        mask = orb_mask.to(E_orb.device).to(err.dtype)
        return (err.pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
