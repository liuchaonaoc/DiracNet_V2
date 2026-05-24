"""Term / level metadata encoding for levelwise (Scheme A) training."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor


class TermVocabulary:
    """Map NIST term strings to integer ids (0 reserved for unknown)."""

    def __init__(self) -> None:
        self._str_to_id: dict[str, int] = {"<unk>": 0}
        self._id_to_str: dict[int, str] = {0: "<unk>"}

    @classmethod
    def from_manifest(cls, df: pd.DataFrame, term_col: str = "term") -> "TermVocabulary":
        vocab = cls()
        if term_col in df.columns:
            for t in df[term_col].astype(str).unique():
                vocab.add(t)
        return vocab

    def add(self, term: str) -> int:
        term = str(term).strip() or "<unk>"
        if term not in self._str_to_id:
            idx = len(self._str_to_id)
            self._str_to_id[term] = idx
            self._id_to_str[idx] = term
        return self._str_to_id[term]

    def encode(self, term: str) -> int:
        return self._str_to_id.get(str(term).strip(), 0)

    def __len__(self) -> int:
        return len(self._str_to_id)

    def id_to_str(self, idx: int) -> str:
        return self._id_to_str.get(int(idx), "<unk>")


class LevelFeatureEncoder(nn.Module):
    """Extra (J, parity, term_id) features fused into h_cond."""

    def __init__(self, d_cond: int, term_vocab_size: int, max_j_2: int = 32) -> None:
        super().__init__()
        self.max_j_2 = max_j_2
        self.term_embed = nn.Embedding(max(term_vocab_size, 1), d_cond, padding_idx=0)
        self.j_embed = nn.Embedding(max_j_2 + 1, d_cond)
        self.parity_embed = nn.Embedding(2, d_cond)
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_cond, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )
        self.ln = nn.LayerNorm(d_cond)

    def forward(self, h_cond: Tensor, J: Tensor, parity: Tensor, term_id: Tensor) -> Tensor:
        J = J.long().clamp(0, self.max_j_2)
        parity = parity.long().clamp(0, 1)
        term_id = term_id.long().clamp(0, self.term_embed.num_embeddings - 1)
        extra = torch.cat([
            self.j_embed(J),
            self.parity_embed(parity),
            self.term_embed(term_id),
        ], dim=-1)
        return self.ln(h_cond + self.fuse(extra))
