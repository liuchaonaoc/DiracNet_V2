"""Orthonormality loss."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class OrthonormalityLoss(nn.Module):
    """‖ S − I ‖_F²."""

    def forward(self, P: Tensor, Q: Tensor, grid, orb_mask: Tensor) -> Tensor:
        integrand = P.unsqueeze(2) * P.unsqueeze(1) + Q.unsqueeze(2) * Q.unsqueeze(1)
        S = grid.integrate(integrand, dim=-1)                  # [B, N_orb, N_orb]
        N_orb = S.shape[-1]
        eye = torch.eye(N_orb, device=S.device, dtype=S.dtype).unsqueeze(0)
        mask_2d = orb_mask.unsqueeze(-1) & orb_mask.unsqueeze(-2)
        diff = (S - eye) * mask_2d.to(S.dtype)
        return (diff ** 2).sum() / mask_2d.sum().clamp_min(1.0).to(S.dtype)
