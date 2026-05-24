"""Warmup + cosine annealing scheduler."""

from __future__ import annotations

import math

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    name = cfg.scheduler.name.lower()
    if name == "warmup_cosine":
        warmup = int(cfg.scheduler.warmup_steps)
        total = int(cfg.scheduler.total_steps)
        min_ratio = float(getattr(cfg.scheduler, "min_lr_ratio", 0.0))
        min_ratio = max(0.0, min(1.0, min_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cos

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    raise ValueError(f"unknown scheduler {name}")
