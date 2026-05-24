"""Physical constants in atomic units (Hartree).

对应 frame_RC_V1_1.md §4.1。
"""

from __future__ import annotations

import math

import torch

ALPHA: float = 1.0 / 137.035999084  # fine-structure constant
C_LIGHT: float = 1.0 / ALPHA        # speed of light in a.u. ≈ 137.036
BOHR_RADIUS_M: float = 5.29177210903e-11
HARTREE_eV: float = 27.211386245988
RYDBERG_eV: float = HARTREE_eV / 2.0
ELECTRON_MASS_AU: float = 1.0
ELEMENTARY_CHARGE_AU: float = 1.0
HBAR_AU: float = 1.0

# 角动量/旋量常量
TWO_PI: float = 2.0 * math.pi


def hartree_to_ev(e: torch.Tensor) -> torch.Tensor:
    """Hartree → eV."""
    return e * HARTREE_eV


def ev_to_hartree(e: torch.Tensor) -> torch.Tensor:
    """eV → Hartree."""
    return e / HARTREE_eV


def cm_inv_to_hartree(e: torch.Tensor) -> torch.Tensor:
    """波数 (cm^-1) → Hartree."""
    return e * 4.5563352529120e-6  # 1 / 219474.6313705
