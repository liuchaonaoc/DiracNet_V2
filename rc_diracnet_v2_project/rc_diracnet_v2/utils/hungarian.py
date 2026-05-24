"""Hungarian (Kuhn-Munkres) matching between predicted and NIST energies.

对应 frame_RC_V1_1.md §4.2 hungarian.py。
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .numeric import PRED_ACTIVE_ABS_MAX_Ha, PRED_ACTIVE_THRESHOLD_Ha


@torch.no_grad()
def prediction_active_mask(
    E_pred: Tensor,
    threshold: float = PRED_ACTIVE_THRESHOLD_Ha,
    abs_max_ha: float = PRED_ACTIVE_ABS_MAX_Ha,
) -> Tensor:
    """标记可参与匹配的物理预测能级。

    排除：
    * 无效轨道 ``1e3 Ha`` 占位；
    * ``sanitize`` 限幅产生的 ``±5e3 Ha`` 假本征值。
    """
    return (
        torch.isfinite(E_pred)
        & (E_pred < threshold)
        & (E_pred.abs() < abs_max_ha)
    )


@torch.no_grad()
def hungarian_match(
    cost: Tensor,
    valid_mask: Tensor,
    pred_active_mask: Tensor | None = None,
    large_cost: float = 1.0e9,
) -> Tensor:
    """对每个 batch 解线性指派问题，返回置换索引。

    参数
    ----
    cost : Tensor [B, M_pred, M_target]
        预测能级与目标能级的代价（一般用 ``(E_pred - E_target)^2``）。
    valid_mask : Tensor [B, M_target] bool
        目标能级是否有效（NIST 缺失则为 False）。
    pred_active_mask : Tensor [B, M_pred] bool, optional
        若给定，仅在活跃预测行上求解；未匹配到的目标列 ``perm=-1``。
        用于排除 ``1e3 Ha`` 占位本征值参与匹配。
    large_cost : float
        无效目标位置填充的大代价。

    返回
    ----
    perm : LongTensor [B, M_target]
        ``perm[b, j]`` 给出与目标 j 对应的预测下标；``-1`` 表示未匹配。
    """
    if cost.dim() != 3:
        raise ValueError("cost must be [B, M_pred, M_target]")
    B, M_pred, M_target = cost.shape

    cost_np = cost.detach().cpu().numpy().astype(np.float64)
    mask_np = valid_mask.detach().cpu().numpy().astype(bool)
    pred_np = None
    if pred_active_mask is not None:
        pred_np = pred_active_mask.detach().cpu().numpy().astype(bool)

    perms = np.full((B, M_target), -1, dtype=np.int64)
    for b in range(B):
        t_cols = np.where(mask_np[b])[0]
        if len(t_cols) == 0:
            continue

        if pred_np is not None:
            p_rows = np.where(pred_np[b])[0]
        else:
            p_rows = np.arange(M_pred)

        if len(p_rows) == 0:
            continue

        c_sub = cost_np[b][np.ix_(p_rows, t_cols)].copy()
        if not np.isfinite(c_sub).all():
            c_sub = np.nan_to_num(c_sub, nan=large_cost, posinf=large_cost, neginf=large_cost)

        n_assign = min(len(p_rows), len(t_cols))
        if n_assign == 0:
            continue

        row_ind, col_ind = linear_sum_assignment(c_sub)
        for r_i, c_i in zip(row_ind, col_ind):
            perms[b, t_cols[c_i]] = int(p_rows[r_i])

    return torch.from_numpy(perms).to(cost.device)


def gather_by_perm(E_pred: Tensor, perm: Tensor, fill: float = float("nan")) -> Tensor:
    """根据 ``perm[b, j]`` 重排 ``E_pred[b, :]``；``perm=-1`` 位置填 ``fill``。

    参数
    ----
    E_pred : Tensor [B, M_pred]
    perm   : LongTensor [B, M_target]，未匹配为 -1

    返回
    ----
    Tensor [B, M_target]
    """
    if E_pred.dim() != 2:
        raise ValueError("E_pred must be [B, M_pred]")
    safe_perm = perm.clamp(min=0)
    matched = torch.gather(E_pred, dim=1, index=safe_perm)
    if (perm < 0).any():
        matched = torch.where(perm >= 0, matched, torch.full_like(matched, fill))
    return matched
