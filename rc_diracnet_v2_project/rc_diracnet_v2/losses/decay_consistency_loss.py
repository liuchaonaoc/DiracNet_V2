"""Asymptotic decay-rate / energy self-consistency loss.

Physics
-------
For any bound state of a single-particle Schrödinger-like equation with a
sufficiently fast-decaying central potential V(r) -> 0 as r -> infinity, the
large-r asymptotic form of the radial wavefunction satisfies

    P(r) ~ r^{...} * exp(- lambda * r),
    lambda = sqrt(- 2 * E_orb)   (a.u., non-relativistic).

In the non-relativistic regime relevant here (|E_orb| << c^2), the same
relation holds for the Dirac large component to leading order.

We use this as a model-free consistency constraint between the network's two
independent outputs (lambda and E_orb-via-Rayleigh-quotient). It contains no
reference to the form of the potential and no reference to target energies.

Note
----
- We penalise the squared residual ``(lambda^2 + 2 * E_orb)`` so that a
  positive ``E_orb`` (an unbound prediction) is also penalised; bound states
  satisfy the constraint exactly.
- Symmetrically applied to all orbitals via ``orb_mask``.
- ADR-017 — Per-sample 相对化归一化 (slice 2 入口)
    ``residual = lam² + 2·E_orb`` 带能量量纲 ``[E]``，平方后 ``[E²]``。
    ``forward`` 接受可选的 ``e_char_sq`` (``[B, N_orb]``，量纲 ``[E²]``)；
    若给定，per-orb ``residual.pow(2)`` 除以 ``e_char_sq.detach()``
    得到无量纲量再 mask-mean。这与 ``DiracPDELoss`` / ``VirialLoss``
    使用同一份 ``e_char_sq``，保证三者在 backward 中量级齐平。
    ``e_char_sq=None`` 时保持 ADR-017 之前的原行为。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DecayConsistencyLoss(nn.Module):
    def forward(
        self,
        lam: Tensor,
        E_orb: Tensor,
        orb_mask: Tensor,
        e_char_sq: Tensor | None = None,    # ADR-017 normalisation
    ) -> Tensor:
        # lam, E_orb: [B, N_orb]; orb_mask: [B, N_orb]
        residual = lam.pow(2) + 2.0 * E_orb
        per_orb = residual.pow(2)
        if e_char_sq is not None:
            # residual ~ [E], per_orb ~ [E²]; e_char_sq ~ [E²] → dimensionless.
            per_orb = per_orb / e_char_sq.detach().clamp_min(1.0e-12)
        mask = orb_mask.to(per_orb.device).to(per_orb.dtype)
        return (per_orb * mask).sum() / mask.sum().clamp_min(1.0)
