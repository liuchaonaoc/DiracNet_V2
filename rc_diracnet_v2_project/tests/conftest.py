from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from rc_diracnet_v2.utils.grid import RadialGrid
from rc_diracnet_v2.basis import BSplineBasis


@pytest.fixture
def default_grid():
    return RadialGrid(r_min=1e-4, r_max=50.0, n_grid=256, scheme="loglinear")


@pytest.fixture
def default_bspline(default_grid):
    bs = BSplineBasis(n_basis=32, order=3, r_min=1e-4, r_max=50.0)
    bs.precompute(default_grid.r)
    return bs


@pytest.fixture
def hydrogenic_h_1s_batch():
    return {
        "Z": torch.tensor([1]), "charge": torch.tensor([0]), "nele": torch.tensor([1]),
        "config_shells": torch.tensor([[[1,0,1,1]] + [[0,0,0,0]]*31]).long(),
        "shell_mask": torch.tensor([[True] + [False]*31]),
        "kappa": torch.tensor([[-1] + [0]*15]),
        "orb_mask": torch.tensor([[True] + [False]*15]),
        "J": torch.tensor([1]), "parity": torch.tensor([0]), "term_id": torch.tensor([0]),
        "E_target": torch.tensor([-0.5]),
    }
