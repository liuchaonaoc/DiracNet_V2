"""Differentiable node-count loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _approx_node_count(P: Tensor) -> Tensor:
    sigma = P.abs().median(dim=-1, keepdim=True).values.clamp_min(1e-6) * 1e-2
    s = torch.tanh(P / sigma)
    return 0.5 * (s[..., 1:] - s[..., :-1]).abs().sum(dim=-1)


class NodeCountLoss(nn.Module):
    def forward(self, P: Tensor, n_required: Tensor, orb_mask: Tensor) -> Tensor:
        n_pred = _approx_node_count(P)
        err = n_pred - n_required.to(P.device).to(n_pred.dtype).clamp_min(0.0)
        mask = orb_mask.to(P.device).to(err.dtype)
        return F.huber_loss(err * mask, torch.zeros_like(err), delta=0.5, reduction="sum") / mask.sum().clamp_min(1.0)
