"""Generalised virial-theorem loss.

Physics
-------
For a single particle in a central potential V(r) (any V, no Coulomb-specific
form assumed), every bound stationary state ψ obeys

    2 <T> = < r * dV/dr >.

We extract <T> from the Rayleigh-quotient energy E_orb as

    <T> = E_orb - <V>,

so the constraint becomes a model-free residual

    R = 2 (E_orb - <V>) - <r * dV/dr> = 0.

This is the generalised virial theorem; the Coulomb-specific form
``2<T> + <V> = 0`` is recovered automatically when V(r) = -Z/r because
``r * dV/dr = -V`` in that case.

Notes
-----
- ``dV/dr`` is computed with ``torch.gradient`` on the non-uniform grid r;
  no analytic form of V is required.
- Per-orbital integrals are normalised by the same orbital density norm so
  the residual is in energy units, comparable across orbitals.
- ADR-017 — Per-sample 相对化归一化 (slice 2 入口)
    ``residual = 2T - <r·V'>`` is in energy units ``[E]``; its square is
    ``[E²]``. When the batch mixes ``(Z=3, n=1)`` (|E|≈4.5 Ha) with
    ``(Z=1, n=4)`` (|E|≈0.03 Ha) the absolute residual² differs by 22 500×,
    so ``.mean()`` lets the deep states dominate the gradient. ``forward``
    accepts an optional ``e_char_sq`` (``[B, N_orb]``, units ``[E²]``); when
    given, per-orbital ``residual.pow(2)`` is divided by
    ``e_char_sq.detach()`` before mask-mean, giving a dimensionless penalty
    that puts every ``(Z, n)`` on the same gradient footing.
    ``e_char_sq=None`` reproduces the pre-ADR-017 behaviour.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VirialLoss(nn.Module):
    def forward(
        self,
        P: Tensor,         # [B, N_orb, N_grid]
        Q: Tensor,
        E_orb: Tensor,     # [B, N_orb]
        V_eff: Tensor,     # [B, N_grid]
        r: Tensor,         # [N_grid]
        grid,
        orb_mask: Tensor,  # [B, N_orb]
        e_char_sq: Tensor | None = None,   # ADR-017 normalisation
    ) -> Tensor:
        rho = P * P + Q * Q                                   # [B, N_orb, N_grid]
        V_b = V_eff.unsqueeze(1).to(rho.dtype)                # [B, 1, N_grid]
        norm = grid.integrate(rho, dim=-1).clamp_min(1.0e-12) # [B, N_orb]

        V_avg = grid.integrate(rho * V_b, dim=-1) / norm      # [B, N_orb]

        # Generalised virial uses r * dV/dr; finite-difference on the
        # (possibly non-uniform) grid keeps the loss potential-agnostic.
        dV_dr = torch.gradient(V_eff.to(rho.dtype), spacing=(r.to(rho.dtype),), dim=-1)[0]
        rdVdr = (r.view(1, -1).to(rho.dtype) * dV_dr).unsqueeze(1)
        rdV_avg = grid.integrate(rho * rdVdr, dim=-1) / norm  # [B, N_orb]

        T_avg = E_orb - V_avg                                 # [B, N_orb]
        residual = 2.0 * T_avg - rdV_avg
        per_orb = residual.pow(2)
        if e_char_sq is not None:
            per_orb = per_orb / e_char_sq.detach().clamp_min(1.0e-12)
        mask = orb_mask.to(per_orb.device).to(per_orb.dtype)
        return (per_orb * mask).sum() / mask.sum().clamp_min(1.0)
