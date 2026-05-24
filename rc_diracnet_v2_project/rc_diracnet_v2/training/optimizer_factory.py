"""Optimizer factory for V2."""

from __future__ import annotations

import torch


def build_optimizer(model, cfg, stage: int = 1):
    if stage == 2 and getattr(model, "residual_head", None) is not None:
        params = [p for p in model.residual_head.parameters() if p.requires_grad]
        lr = float(cfg.optimizer.lr_residual)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        lr = float(cfg.optimizer.lr_kan)
    return torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.optimizer.weight_decay))
