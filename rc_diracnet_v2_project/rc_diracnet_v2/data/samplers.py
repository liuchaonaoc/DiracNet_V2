"""Batch samplers for levelwise training (group by ion)."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class GroupByIonSampler(Sampler[list[int]]):
    """Yield batches of indices sharing the same (Z, ion_charge).

    Ensures ``align_energies_to_ground`` has ≥2 levels per ion when possible.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: int = 0) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self._groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        df = dataset.df
        for idx in range(len(dataset)):
            row = df.iloc[idx]
            key = (int(row["Z"]), int(row["ion_charge"]))
            self._groups[key].append(idx)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        batches: list[list[int]] = []
        group_keys = list(self._groups.keys())
        if self.shuffle:
            perm = torch.randperm(len(group_keys), generator=g).tolist()
            group_keys = [group_keys[i] for i in perm]

        for key in group_keys:
            idxs = list(self._groups[key])
            if self.shuffle:
                perm_i = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm_i]
            for start in range(0, len(idxs), self.batch_size):
                batches.append(idxs[start : start + self.batch_size])

        if self.shuffle:
            perm_b = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm_b]

        yield from batches

    def __len__(self) -> int:
        n = 0
        for idxs in self._groups.values():
            n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n
