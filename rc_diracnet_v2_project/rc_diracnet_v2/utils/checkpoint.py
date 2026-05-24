"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path, is_best: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "best.pt"
        torch.save(state, best_path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    # PyTorch >=2.6 defaults weights_only=True; our checkpoints embed OmegaConf cfg.
    return torch.load(Path(path), map_location=map_location, weights_only=False)
