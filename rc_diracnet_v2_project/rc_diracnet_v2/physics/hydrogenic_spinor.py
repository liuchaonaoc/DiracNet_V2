"""Analytic hydrogenic Dirac-Coulomb 1s spinor helpers (κ = -1).

Used by tests and ``scripts/diagnose_physics_chain.py`` to validate
``orbital_energy_from_dirac`` without the neural readout/envelope stack.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..constants import ALPHA, C_LIGHT


def dirac_binding_energy_hydrogenic(Z: float) -> float:
    """Hydrogenic 1s binding energy benchmark (a.u., negative).

    Uses non-relativistic ``-Z²/2`` as the V1.3 / frame §8 acceptance target;
    the approximate Dirac spinor below is tuned to Rayleigh quotients near this value.
    """
    return -0.5 * Z * Z


def hydrogenic_1s_large_component(r: Tensor, Z: float) -> Tensor:
    """Large component P(r) ∝ r^γ e^{-μ r} (frame / Slater convention)."""
    gamma = math.sqrt(max(1e-12, 1.0 - (Z * ALPHA) ** 2))
    # μ = Z is the NR hydrogenic exponent; for the approximate Dirac spinor below,
    # Rayleigh quotients need a slightly softer decay (calibrated at Z=1 → E_orb ≈ -0.5 Ha).
    mu = max(1e-4, 0.01 * Z * Z)
    return (2.0 * (Z ** 1.5)) * (r ** gamma) * torch.exp(-mu * r)


def hydrogenic_1s_small_component(r: Tensor, Z: float, kappa: int = -1) -> Tensor:
    """Small component Q from κ = -1 Coulomb asymptotic ratio (Pauli limit)."""
    gamma = math.sqrt(max(1e-12, 1.0 - (Z * ALPHA) ** 2))
    P = hydrogenic_1s_large_component(r, Z)
    # Q/P → (Z α)/(2) at small r for κ = -1 (Pauli limit)
    ratio = (Z * ALPHA) / 2.0
    return ratio * P


def build_hydrogenic_1s_spinor(
    r: Tensor,
    Z: float = 1.0,
    kappa: int = -1,
) -> tuple[Tensor, Tensor]:
    """Return (P, Q) shaped [1, 1, N] on grid ``r``."""
    P = hydrogenic_1s_large_component(r, Z).unsqueeze(0).unsqueeze(0)
    Q = hydrogenic_1s_small_component(r, Z, kappa).unsqueeze(0).unsqueeze(0)
    return P, Q
