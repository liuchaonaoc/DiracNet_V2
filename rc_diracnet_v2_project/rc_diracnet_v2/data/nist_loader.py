"""NIST level loader.

对应 frame_RC_V1_1.md §4.4 nist_loader.py。

支持两种数据源：
    * NIST ASD CSV/Parquet 文件（清洗后由 ``scripts/prepare_nist_dataset.py`` 生成）；
    * 内置合成 hydrogenic 数据集（首次运行无 NIST 时由 ``synthesize_hydrogenic``
      生成）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..constants import HARTREE_eV


REQUIRED_COLUMNS = [
    "Z", "ion_charge", "parent_config", "level_config",
    "J", "parity", "level_eV", "uncertainty_eV", "term",
]


class NISTLevelLoader:
    """加载清洗后的 NIST 能级 manifest。"""

    def __init__(self, source: str | Path) -> None:
        self.path = Path(source)
        if not self.path.exists():
            raise FileNotFoundError(f"NIST manifest not found: {self.path}")
        if self.path.suffix == ".parquet":
            self.df = pd.read_parquet(self.path)
        else:
            self.df = pd.read_csv(self.path)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"NIST manifest missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def unique_parents(self) -> pd.DataFrame:
        """对外暴露每个 (Z, ion_charge, parent_config) 的代表行。"""
        return self.df[["Z", "ion_charge", "parent_config"]].drop_duplicates().reset_index(drop=True)

    def query(self, Z: int, ion_charge: int, parent_config: str) -> pd.DataFrame:
        mask = (
            (self.df["Z"] == Z)
            & (self.df["ion_charge"] == ion_charge)
            & (self.df["parent_config"] == parent_config)
        )
        return self.df.loc[mask].sort_values("level_eV").reset_index(drop=True)

    @staticmethod
    def to_target_tensor(
        df: pd.DataFrame, max_M: int,
    ) -> tuple[Tensor, Tensor]:
        """返回 (E_target [max_M] Hartree, valid_mask [max_M])。"""
        e_target = torch.zeros(max_M, dtype=torch.float32)
        mask = torch.zeros(max_M, dtype=torch.bool)
        if len(df) == 0:
            return e_target, mask
        levels = df["level_eV"].to_numpy()
        finite = np.isfinite(levels)
        levels = levels[finite]
        n_use = min(len(levels), max_M)
        if n_use > 0:
            e_target[:n_use] = torch.tensor(
                levels[:n_use] / HARTREE_eV, dtype=torch.float32
            )
            mask[:n_use] = True
        return e_target, mask
