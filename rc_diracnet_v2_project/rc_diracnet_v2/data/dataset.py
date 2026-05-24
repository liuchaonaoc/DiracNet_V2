"""AtomicSpectraDataset + collate fn.

对应 frame_RC_V1_1.md §4.4 dataset.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from ..constants import HARTREE_eV
from .config_parser import encode_config_to_tensor, parse_config_string
from .level_encoder import TermVocabulary
from .nist_loader import NISTLevelLoader


class AtomicSpectraDataset(Dataset):
    """每个样本 = (Z, ion_charge, parent_config) + 该原子下所有 NIST 能级目标。"""

    def __init__(
        self,
        manifest_path: str | Path,
        max_orb: int = 16,
        max_seq: int = 16,
        max_target_M: int = 32,
    ) -> None:
        super().__init__()
        self.loader = NISTLevelLoader(manifest_path)
        self.parents = self.loader.unique_parents()
        self.max_orb = max_orb
        self.max_seq = max_seq
        self.max_target_M = max_target_M

    def __len__(self) -> int:
        return len(self.parents)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.parents.iloc[idx]
        Z = int(row["Z"])
        ion_charge = int(row["ion_charge"])
        parent_config = str(row["parent_config"])

        df = self.loader.query(Z, ion_charge, parent_config)
        e_target, e_mask = NISTLevelLoader.to_target_tensor(df, self.max_target_M)

        shells = parse_config_string(parent_config)
        config_tensor = encode_config_to_tensor(shells, self.max_seq)
        n_orb = min(len(shells), self.max_orb)

        nele = sum(s.occ for s in shells)
        # parity = (-1)^(sum l*occ)
        parity = sum(s.l * s.occ for s in shells) % 2

        # κ 序列（用于 envelope / dirac operator）
        kappa = torch.zeros(self.max_orb, dtype=torch.long)
        for i, sh in enumerate(shells[: self.max_orb]):
            kappa[i] = sh.kappa
        orb_mask = torch.zeros(self.max_orb, dtype=torch.bool)
        orb_mask[:n_orb] = True

        # J、parity 标签来自 NIST（每个目标态可以不同；这里给 batch-level parent 标签）
        J_parent = int(2 * df["J"].iloc[0]) if len(df) > 0 else 0  # 2J 整数

        sample: dict[str, Any] = {
            "Z": torch.tensor(Z, dtype=torch.long),
            "nele": torch.tensor(nele, dtype=torch.long),
            "charge": torch.tensor(ion_charge, dtype=torch.long),
            "J": torch.tensor(J_parent, dtype=torch.long),
            "parity": torch.tensor(parity, dtype=torch.long),
            "config_shells": config_tensor,                  # [max_seq, 4]
            "shell_mask": (config_tensor.sum(dim=-1) > 0),   # [max_seq]
            "kappa": kappa,                                  # [max_orb]
            "orb_mask": orb_mask,                            # [max_orb]
            "E_target": e_target,                            # [max_target_M] Hartree
            "E_mask": e_mask,                                # [max_target_M]
            "parent_config_str": parent_config,
        }
        return sample


class LevelRowDataset(Dataset):
    """方案 A：manifest 每一行 = 一个样本，预测标量能级。

    使用 ``level_config`` 解析壳层；``E_target`` 为 Hartree（由 ``level_eV`` 转换）。
    激发能相对同一 (Z, ion_charge) 的最低能级归零（与 NIST 惯例一致）。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        max_orb: int = 16,
        max_seq: int = 16,
        term_vocab: TermVocabulary | None = None,
        align_in_loader: bool = True,
    ) -> None:
        super().__init__()
        self.loader = NISTLevelLoader(manifest_path)
        self.df = self.loader.df.copy()
        self.max_orb = max_orb
        self.max_seq = max_seq
        self.term_vocab = term_vocab or TermVocabulary.from_manifest(self.df)
        self.align_in_loader = bool(align_in_loader)

        if self.align_in_loader:
            # excitation energies relative to ion ground (baseline behaviour)
            self.df["level_eV"] = self.df.groupby(["Z", "ion_charge"])["level_eV"].transform(
                lambda s: s - s.min()
            )
        finite = np.isfinite(self.df["level_eV"].to_numpy())
        self.df = self.df.loc[finite].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        Z = int(row["Z"])
        ion_charge = int(row["ion_charge"])
        level_config = str(row["level_config"])

        shells = parse_config_string(level_config)
        config_tensor = encode_config_to_tensor(shells, self.max_seq)
        n_orb = min(len(shells), self.max_orb)
        nele = sum(s.occ for s in shells)
        parity = int(row["parity"]) if "parity" in row else sum(s.l * s.occ for s in shells) % 2
        J_2 = int(round(2.0 * float(row["J"])))
        term = str(row.get("term", ""))

        kappa = torch.zeros(self.max_orb, dtype=torch.long)
        for i, sh in enumerate(shells[: self.max_orb]):
            kappa[i] = sh.kappa
        orb_mask = torch.zeros(self.max_orb, dtype=torch.bool)
        orb_mask[:n_orb] = True

        e_ha = float(row["level_eV"]) / HARTREE_eV

        return {
            "Z": torch.tensor(Z, dtype=torch.long),
            "nele": torch.tensor(nele, dtype=torch.long),
            "charge": torch.tensor(ion_charge, dtype=torch.long),
            "J": torch.tensor(J_2, dtype=torch.long),
            "parity": torch.tensor(parity, dtype=torch.long),
            "term_id": torch.tensor(self.term_vocab.encode(term), dtype=torch.long),
            "config_shells": config_tensor,
            "shell_mask": (config_tensor.sum(dim=-1) > 0),
            "kappa": kappa,
            "orb_mask": orb_mask,
            "E_target": torch.tensor(e_ha, dtype=torch.float32),
            "level_config_str": level_config,
            "parent_config_str": level_config,
        }


def collate_levelwise(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack levelwise samples; ``E_target`` is [B] scalar."""
    if not samples:
        raise ValueError("collate_levelwise got empty batch")
    out: dict[str, Any] = {}
    for k in samples[0].keys():
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


def collate_atoms(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """简单 stack；所有样本已经被 padding 到统一形状。"""
    if not samples:
        raise ValueError("collate_atoms got empty batch")

    out: dict[str, Any] = {}
    for k in samples[0].keys():
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out
