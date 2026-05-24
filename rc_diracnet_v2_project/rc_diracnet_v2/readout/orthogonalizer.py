"""Löwdin symmetric orthonormalization (differentiable).

对应 frame_RC_V1_1.md §4.7 orthogonalizer.py。
"""

from __future__ import annotations

import torch
from torch import Tensor


def lowdin_orthonormalize(
    P: Tensor, Q: Tensor,
    grid,  # rc_diracnet.utils.grid.RadialGrid
    orb_mask: Tensor,
    dPdr: Tensor | None = None,
    dQdr: Tensor | None = None,
    eps: float = 1.0e-6,
    use_full_lowdin: bool = False,
) -> dict[str, Tensor]:
    """正交归一化包装。

    默认（``use_full_lowdin=False``）只做 per-orbital L2 归一化，
    通过 ``OrthonormalityLoss`` 软约束跨轨道正交性。
    这样可避免 ``torch.linalg.eigh`` 在初始训练阶段（重叠矩阵接近退化）
    出现 NaN 梯度的著名问题。

    将 ``use_full_lowdin=True`` 时启用完整对称正交化（适合后期微调）。

    输入
    ----
    P, Q     : [B, N_orb, N_grid]
    orb_mask : [B, N_orb] bool
    dPdr/dQdr: 可选；若提供则同步变换。
    """
    if P.shape != Q.shape:
        raise ValueError("P, Q must have same shape")
    B, N_orb, _ = P.shape

    # Step-1: 单轨道归一化（始终执行）
    norm_sq = grid.integrate(P * P + Q * Q, dim=-1).clamp_min(eps)        # [B, N_orb]
    inv_norm = norm_sq.rsqrt().unsqueeze(-1)                              # [B, N_orb, 1]
    P = P * inv_norm
    Q = Q * inv_norm
    if dPdr is not None:
        dPdr = dPdr * inv_norm
    if dQdr is not None:
        dQdr = dQdr * inv_norm

    out: dict[str, Tensor] = {"P": P, "Q": Q}
    if dPdr is not None:
        out["dPdr"] = dPdr
    if dQdr is not None:
        out["dQdr"] = dQdr

    if not use_full_lowdin:
        return out

    # Step-2 (可选): 完整 Löwdin
    integrand = P.unsqueeze(2) * P.unsqueeze(1) + Q.unsqueeze(2) * Q.unsqueeze(1)
    S = grid.integrate(integrand, dim=-1)
    mask_2d = orb_mask.unsqueeze(-1) & orb_mask.unsqueeze(-2)
    eye = torch.eye(N_orb, device=S.device, dtype=S.dtype).unsqueeze(0).expand_as(S)
    S = torch.where(mask_2d, S, eye)
    S = 0.5 * (S + S.transpose(-1, -2)) + eps * eye
    eigvals, eigvecs = torch.linalg.eigh(S)
    inv_sqrt = eigvecs @ torch.diag_embed(eigvals.clamp_min(eps).rsqrt()) @ eigvecs.transpose(-1, -2)
    out["P"] = torch.einsum("bac,bcg->bag", inv_sqrt, P)
    out["Q"] = torch.einsum("bac,bcg->bag", inv_sqrt, Q)
    if dPdr is not None:
        out["dPdr"] = torch.einsum("bac,bcg->bag", inv_sqrt, dPdr)
    if dQdr is not None:
        out["dQdr"] = torch.einsum("bac,bcg->bag", inv_sqrt, dQdr)
    return out
