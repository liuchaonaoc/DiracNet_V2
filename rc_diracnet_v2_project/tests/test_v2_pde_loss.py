"""Tests for ``DiracPDELoss`` — ADR-017 per-sample normalisation surface.

The PDE residual is the largest energy-dimensioned loss in the training mix,
so it sees the biggest absolute scale spread across (Z, n). These tests pin
down the ADR-017 normalisation behaviour:

1. ``e_char_sq=None`` reproduces the pre-ADR-017 behaviour exactly.
2. With a uniform ``e_char_sq`` the normalised loss equals raw / e_char_sq.
3. With a varying ``e_char_sq`` per orbital, the normalisation is applied
   per-(B, N_orb) before mask-mean.
4. The normalisation flows the gradient correctly (no double-detach bug).
"""

from __future__ import annotations

import torch

from rc_diracnet_v2.losses import DiracPDELoss
from rc_diracnet_v2.utils.grid import RadialGrid


def _make_smooth_state(grid: RadialGrid, n_orb: int = 2):
    """Build a synthetic two-orbital batch with smooth P/Q (no real eigenstate
    needed — we just want to verify the loss arithmetic, not the physics)."""
    r = grid.r.to(torch.float32)
    P = torch.zeros(1, n_orb, r.numel(), dtype=torch.float32)
    Q = torch.zeros_like(P)
    # Different amplitudes to make the (B, N_orb) dim non-trivial.
    P[0, 0] = 2.0 * r * torch.exp(-r)
    P[0, 1] = r * torch.exp(-0.5 * r)
    # dP/dr by hand for the synthetic profiles.
    dPdr = torch.zeros_like(P)
    dPdr[0, 0] = (2.0 - 2.0 * r) * torch.exp(-r)
    dPdr[0, 1] = (1.0 - 0.5 * r) * torch.exp(-0.5 * r)
    dQdr = torch.zeros_like(Q)
    E_orb = torch.tensor([[-0.5, -0.125]], dtype=torch.float32)
    V_eff = (-1.0 / r).view(1, -1)
    kappa = torch.tensor([[-1, -1]], dtype=torch.int64)
    orb_mask = torch.tensor([[True, True]])
    return P, Q, dPdr, dQdr, E_orb, V_eff, kappa, orb_mask


def test_pde_e_char_sq_none_matches_no_arg():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = DiracPDELoss(detach_energy=True)
    P, Q, dPdr, dQdr, E, V, kappa, mask = _make_smooth_state(grid)
    a = loss_fn(P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask).item()
    b = loss_fn(P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask, e_char_sq=None).item()
    assert a == b


def test_pde_uniform_e_char_sq_equals_raw_over_constant():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = DiracPDELoss(detach_energy=True)
    P, Q, dPdr, dQdr, E, V, kappa, mask = _make_smooth_state(grid)
    raw = loss_fn(P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask).item()
    uniform = 0.0625
    e_char_sq = torch.full_like(E, uniform)
    normalised = loss_fn(
        P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask, e_char_sq=e_char_sq
    ).item()
    assert abs(normalised - raw / uniform) < 1.0e-5


def test_pde_per_orbital_e_char_sq_applied_before_mean():
    """When e_char_sq differs across orbitals, the loss must divide per-orb
    BEFORE mask-mean — otherwise the deep orbital dominates the average."""
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = DiracPDELoss(detach_energy=True)
    P, Q, dPdr, dQdr, E, V, kappa, mask = _make_smooth_state(grid)
    # Two raw residuals first
    raw_total = loss_fn(P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask).item()
    # Construct an e_char_sq that strongly attenuates orbital 0 (the deep one).
    # If normalisation were applied AFTER mean, this would only change the
    # output by a constant; applied per-orb-before-mean, the relative
    # weighting changes.
    e_uniform = torch.full_like(E, 1.0)
    e_strong_first = torch.tensor([[100.0, 1.0]], dtype=E.dtype)
    val_uniform = loss_fn(
        P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask, e_char_sq=e_uniform
    ).item()
    val_strong = loss_fn(
        P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask, e_char_sq=e_strong_first
    ).item()
    # val_uniform == raw_total (uniform=1 division is identity).
    assert abs(val_uniform - raw_total) < 1.0e-6
    # val_strong should down-weight orbital 0 by ~100×; orbital 1 unchanged.
    # The new mean is roughly (raw_0/100 + raw_1) / 2; orbital 0 is the deep
    # one with the bigger residual (~bigger pre-norm value), so val_strong
    # must be markedly smaller than val_uniform.
    assert val_strong < val_uniform


def test_pde_gradient_flows_through_normalised_loss():
    """E_char² is detached internally; gradient should still flow through P."""
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = DiracPDELoss(detach_energy=True)
    P, Q, dPdr, dQdr, E, V, kappa, mask = _make_smooth_state(grid)
    P = P.detach().clone().requires_grad_(True)
    dPdr = dPdr.detach().clone().requires_grad_(True)
    e_char_sq = torch.tensor([[0.5, 0.05]], dtype=E.dtype)
    val = loss_fn(P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask, e_char_sq=e_char_sq)
    val.backward()
    assert P.grad is not None and torch.any(P.grad.abs() > 0)


def test_pde_e_char_sq_floor_does_not_explode_when_lam_tiny():
    """If a caller passes a near-zero e_char_sq, the internal clamp_min(1e-12)
    must keep the loss finite (the trainer-level helper would normally floor
    earlier, but the loss itself is the final safety net)."""
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=128, scheme="loglinear")
    loss_fn = DiracPDELoss(detach_energy=True)
    P, Q, dPdr, dQdr, E, V, kappa, mask = _make_smooth_state(grid)
    e_char_sq = torch.zeros_like(E)
    val = loss_fn(P, Q, dPdr, dQdr, E, V, kappa, grid.r, grid, mask, e_char_sq=e_char_sq).item()
    assert torch.isfinite(torch.tensor(val))
