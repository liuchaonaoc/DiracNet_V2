"""Scalar NIST Huber loss with per-ion ground-state alignment (Scheme A)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.numeric import sanitize_energies


def align_energies_to_ground(
    E: Tensor,
    Z: Tensor,
    charge: Tensor,
) -> Tensor:
    """Per (Z, ion_charge) subtract batch-local minimum (excitation energies)."""
    out = E.clone()
    keys = Z.long() * 10000 + charge.long()
    for key in torch.unique(keys):
        mask = keys == key
        out[mask] = E[mask] - E[mask].min()
    return out


class NISTScalarHuberLoss(nn.Module):
    """Huber on scalar predictions; optional ground alignment per ion in batch."""

    def __init__(self, delta: float = 0.05, align_ground: bool = True) -> None:
        super().__init__()
        self.delta = delta
        self.align_ground = align_ground

    def forward(
        self,
        E_pred: Tensor,
        E_target: Tensor,
        Z: Tensor,
        charge: Tensor,
    ) -> Tensor:
        """E_pred, E_target: [B] Hartree."""
        E_pred = sanitize_energies(E_pred.unsqueeze(-1)).squeeze(-1)
        E_tgt = torch.nan_to_num(E_target, nan=0.0)
        finite = torch.isfinite(E_tgt) & torch.isfinite(E_pred)
        if not finite.any():
            return E_pred.sum() * 0.0

        if self.align_ground:
            E_pred = align_energies_to_ground(E_pred, Z, charge)
            E_tgt = align_energies_to_ground(E_tgt, Z, charge)

        diff = (E_pred - E_tgt)[finite]
        return F.huber_loss(diff, torch.zeros_like(diff), delta=self.delta, reduction="mean")
