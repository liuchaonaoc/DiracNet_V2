"""Generic tensor helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def safe_normalize_radial(
    P: Tensor,
    Q: Tensor,
    grid,  # rc_diracnet.utils.grid.RadialGrid
    eps: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """对每个 (B, orb) 计算 ∫ (P²+Q²) dr 并归一化。

    P, Q : [B, N_orb, N_grid]
    """
    rho = P * P + Q * Q
    norm = grid.integrate(rho, dim=-1)             # [B, N_orb]
    scale = 1.0 / torch.sqrt(norm.clamp_min(eps))  # [B, N_orb]
    return P * scale.unsqueeze(-1), Q * scale.unsqueeze(-1)


def mask_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """带掩码的 softmax。"""
    neg_inf = torch.finfo(logits.dtype).min / 2
    logits = logits.masked_fill(~mask, neg_inf)
    return torch.softmax(logits, dim=dim)


def assert_shape(x: Tensor, expected: tuple[int | None, ...], name: str = "x") -> None:
    """形状断言；None 表示任意。"""
    if x.dim() != len(expected):
        raise AssertionError(f"{name}: expected dim {len(expected)} got {x.dim()}, shape {tuple(x.shape)}")
    for i, (exp_d, got_d) in enumerate(zip(expected, x.shape)):
        if exp_d is not None and exp_d != got_d:
            raise AssertionError(f"{name}: dim {i} expected {exp_d} got {got_d}, full {tuple(x.shape)}")
