import torch

from rc_diracnet_v2.physics.hydrogenic_analytic import (
    hydrogenic_P_analytic,
    hydrogenic_lobe_partition,
    hydrogenic_radial_nodes,
)
from rc_diracnet_v2.training.two_stage_trainer import TwoStageTrainer
from rc_diracnet_v2.utils.grid import RadialGrid


def _make_trainer(grid: RadialGrid) -> TwoStageTrainer:
    trainer = object.__new__(TwoStageTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = type("ModelStub", (), {"grid": grid})()
    return trainer


def test_hydrogenic_radial_nodes_2s():
    grid = RadialGrid(r_min=1e-4, r_max=20.0, n_grid=128)
    nodes = hydrogenic_radial_nodes(grid.r, Z=3, n=2, l=0)
    assert nodes.numel() == 1
    assert torch.allclose(nodes[0], torch.tensor(2.0 / 3.0), atol=1e-6)


def test_lobe_partition_2s_has_two_disjoint_lobes():
    grid = RadialGrid(r_min=1e-4, r_max=20.0, n_grid=256)
    mask = hydrogenic_lobe_partition(grid.r, Z=3, n=2, l=0)
    assert mask.shape == (2, grid.r.shape[-1])
    # Lobes must be disjoint and cover every grid point exactly once.
    assert torch.all(mask.sum(dim=0) == 1)
    # Inner lobe ends before the analytic node, outer lobe starts at/after it.
    r_node = 2.0 / 3.0
    assert grid.r[mask[0]].max() < r_node
    assert grid.r[mask[1]].min() >= r_node


def test_lobe_partition_handles_nodeless_orbitals():
    grid = RadialGrid(r_min=1e-4, r_max=20.0, n_grid=128)
    mask = hydrogenic_lobe_partition(grid.r, Z=1, n=1, l=0)
    assert mask.shape == (1, grid.r.shape[-1])
    assert bool(mask.all().item())


def test_node_position_and_sign_losses_are_small_for_analytic_2s():
    grid = RadialGrid(r_min=1e-4, r_max=20.0, n_grid=256)
    trainer = _make_trainer(grid)
    batch = {
        "nele": torch.tensor([1]),
        "Z": torch.tensor([3]),
        "config_shells": torch.tensor([[[2, 0, 1, 1]]]).long(),
    }
    P = hydrogenic_P_analytic(grid.r, Z=3, n=2, l=0).view(1, 1, -1)
    mask = torch.tensor([[True]])
    node_pos = trainer._hydrogenic_node_position_loss(batch, P, mask)
    node_cross = trainer._hydrogenic_node_crossing_loss(batch, P, mask)
    sign = trainer._hydrogenic_sign_loss(batch, P, mask)
    assert node_pos < 1e-2
    assert node_cross < 1e-2
    assert sign < 0.1


def test_lobe_ratio_loss_is_near_zero_for_analytic_2s():
    grid = RadialGrid(r_min=1e-4, r_max=30.0, n_grid=512)
    trainer = _make_trainer(grid)
    batch = {
        "nele": torch.tensor([1]),
        "Z": torch.tensor([3]),
        "config_shells": torch.tensor([[[2, 0, 1, 1]]]).long(),
    }
    P = hydrogenic_P_analytic(grid.r, Z=3, n=2, l=0).view(1, 1, -1)
    mask = torch.tensor([[True]])
    loss = trainer._hydrogenic_lobe_ratio_loss(batch, P, mask)
    assert float(loss) < 1e-4


def test_lobe_ratio_loss_penalises_outer_lobe_suppression():
    grid = RadialGrid(r_min=1e-4, r_max=30.0, n_grid=512)
    trainer = _make_trainer(grid)
    batch = {
        "nele": torch.tensor([1]),
        "Z": torch.tensor([3]),
        "config_shells": torch.tensor([[[2, 0, 1, 1]]]).long(),
    }
    P_ref = hydrogenic_P_analytic(grid.r, Z=3, n=2, l=0)
    # Suppress the outer lobe by 80% — a textbook n-collapse signature.
    seg_mask = hydrogenic_lobe_partition(grid.r, Z=3, n=2, l=0)
    weight = torch.where(seg_mask[1], torch.tensor(0.2), torch.tensor(1.0))
    P_collapsed = (P_ref * weight).view(1, 1, -1)
    mask = torch.tensor([[True]])
    good = trainer._hydrogenic_lobe_ratio_loss(batch, P_ref.view(1, 1, -1), mask)
    bad = trainer._hydrogenic_lobe_ratio_loss(batch, P_collapsed, mask)
    assert float(bad) > 10.0 * float(good) + 0.5


def test_lobe_ratio_loss_is_zero_for_nodeless_orbital():
    grid = RadialGrid(r_min=1e-4, r_max=20.0, n_grid=256)
    trainer = _make_trainer(grid)
    batch = {
        "nele": torch.tensor([1]),
        "Z": torch.tensor([1]),
        "config_shells": torch.tensor([[[1, 0, 1, 1]]]).long(),
    }
    # Even a perturbed 1s should yield exactly zero loss (no nodes ⇒ skipped).
    P = (hydrogenic_P_analytic(grid.r, Z=1, n=1, l=0) * 3.7).view(1, 1, -1)
    mask = torch.tensor([[True]])
    loss = trainer._hydrogenic_lobe_ratio_loss(batch, P, mask)
    assert float(loss) == 0.0
