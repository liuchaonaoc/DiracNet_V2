"""V2 batch builder: attach per-orbital physical features."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class V2BatchBuilder:
    """Add per-orbital physical features for KAN/MLP coefficient networks."""

    def __init__(self, n_orb_max: int = 16) -> None:
        self.n_orb_max = int(n_orb_max)

    def build(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["per_orb_features"] = self._build_per_orb_features(batch)
        return batch

    def _build_per_orb_features(self, batch: dict[str, Any]) -> Tensor:
        config = batch["config_shells"].long()
        Z = batch["Z"].float()
        orb_mask = batch["orb_mask"].bool()
        device = config.device
        Bsz = config.shape[0]
        n_orb = min(self.n_orb_max, config.shape[1], orb_mask.shape[1])

        n_idx = torch.zeros(Bsz, self.n_orb_max, device=device, dtype=torch.float32)
        l_idx = torch.zeros_like(n_idx)
        j2_idx = torch.zeros_like(n_idx)
        occ = torch.zeros_like(n_idx)
        n_idx[:, :n_orb] = config[:, :n_orb, 0].float()
        l_idx[:, :n_orb] = config[:, :n_orb, 1].float()
        j2_idx[:, :n_orb] = config[:, :n_orb, 2].float()
        occ[:, :n_orb] = config[:, :n_orb, 3].float()

        kappa = torch.where(j2_idx == 2.0 * l_idx - 1.0, l_idx, -(l_idx + 1.0))
        z_b = Z.view(-1, 1).expand_as(n_idx)
        # Conservative Slater-like placeholder; KAN can learn residual behaviour.
        z_eff = (z_b - 0.85 * (n_idx - 1.0).clamp_min(0.0)).clamp_min(0.1)
        if "nele" in batch:
            one_electron = batch["nele"].to(device).view(-1, 1).float() <= 1.5
            z_eff = torch.where(one_electron, z_b, z_eff)
        n_star = torch.where(
            n_idx <= 3.0,
            n_idx.clamp_min(1.0),
            torch.where(n_idx == 4.0, torch.full_like(n_idx, 3.7), torch.where(n_idx == 5.0, torch.full_like(n_idx, 4.0), torch.full_like(n_idx, 4.2))),
        )
        is_outer = torch.zeros_like(n_idx)
        for b in range(Bsz):
            active = torch.nonzero(orb_mask[b, : self.n_orb_max], as_tuple=False).flatten()
            if active.numel() > 0:
                is_outer[b, active[-1]] = 1.0

        features = torch.stack([z_b, z_eff, n_idx, n_star, l_idx, kappa, occ, is_outer], dim=-1)
        features = features * orb_mask[:, : self.n_orb_max].unsqueeze(-1).to(features.dtype)
        return features


# Backwards-compatible alias used by some scripts/tests.
BatchBuilder = V2BatchBuilder
