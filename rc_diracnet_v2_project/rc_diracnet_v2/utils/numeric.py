"""数值稳定工具（能级、哈密顿量等）。"""

from __future__ import annotations

import torch
from torch import Tensor

# Hartree 能级合理窗口（用于训练早期抑制 NaN 传播）
DEFAULT_E_LO = -5_000.0
DEFAULT_E_HI = 5_000.0

# 与 rc_diracnet.models.rc_diracnet 中无效轨道占位能级一致（1e3 Ha）
INACTIVE_ORB_ENERGY_Ha = 1.0e3
PRED_ACTIVE_THRESHOLD_Ha = 999.0   # 排除 1e3 Ha 占位槽
PRED_ACTIVE_ABS_MAX_Ha = 100.0     # NIST 激发能量级；排除 sanitize 限幅到 5e3 Ha 的假本征值


def sanitize_energies(
    E: Tensor,
    fallback: Tensor | None = None,
    lo: float = DEFAULT_E_LO,
    hi: float = DEFAULT_E_HI,
    clamp_finite: bool = True,
) -> Tensor:
    """Replace non-finite energies with ``fallback`` and (optionally) clamp.

    ``clamp_finite=False`` keeps finite values untouched so gradients keep
    flowing even when ``E`` lands in the saturation region. The levelwise
    Scheme A path (``RCDiracNetLevelwise``) prefers ``clamp_finite=False``
    because a saturated Rayleigh quotient at init must remain trainable.
    """
    if fallback is None:
        fallback = torch.zeros_like(E)
    out = torch.where(torch.isfinite(E), E, fallback)
    out = torch.nan_to_num(out, nan=0.0, posinf=hi, neginf=lo)
    if clamp_finite:
        out = out.clamp(min=lo, max=hi)
    return out


def sanitize_tensor(
    x: Tensor,
    lo: float | None = None,
    hi: float | None = None,
) -> Tensor:
    """通用有限化：NaN→0，可选限幅。"""
    if hi is not None and lo is not None:
        out = torch.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
        return out.clamp(min=lo, max=hi)
    out = torch.nan_to_num(x, nan=0.0)
    if lo is not None:
        out = out.clamp_min(lo)
    if hi is not None:
        out = out.clamp_max(hi)
    return out
