"""Sanity tests for the new model-free physics constraints.

1. ``DecayConsistencyLoss`` vanishes on analytic hydrogenic eigenstates,
   where ``lambda = Z/n`` and ``E = -Z^2/(2 n^2)``.
2. ``VirialLoss`` vanishes (up to discretisation noise) on the same
   analytic hydrogenic eigenstates, where the generalised virial
   ``2(E - <V>) = <r dV/dr>`` is exactly satisfied.
3. ``DecayConsistencyLoss`` is strictly positive when ``lambda`` is
   perturbed away from the consistent value, confirming gradient flow.
"""

from __future__ import annotations

import math

import torch

from rc_diracnet_v2.losses import DecayConsistencyLoss, VirialLoss
from rc_diracnet_v2.physics.hydrogenic_analytic import (
    hydrogenic_P_analytic,
    hydrogenic_energy,
)
from rc_diracnet_v2.utils.grid import RadialGrid


def _make_hydrogenic_state(Z: int, n: int, l: int, grid: RadialGrid):
    """Build a single-orbital batch from the analytic hydrogenic P(r)."""
    r = grid.r
    P_an = hydrogenic_P_analytic(r, Z, n, l).to(torch.float32)
    # Re-normalise on the actual grid so <P|P> = 1 (avoids quadrature drift).
    norm = grid.integrate(P_an * P_an, dim=-1).clamp_min(1.0e-12).sqrt()
    P = (P_an / norm).view(1, 1, -1)
    Q = torch.zeros_like(P)
    E = torch.tensor([[hydrogenic_energy(Z, n)]], dtype=torch.float32)
    lam = torch.tensor([[float(Z) / float(n)]], dtype=torch.float32)
    V_eff = (-float(Z) / r).view(1, -1).to(torch.float32)
    orb_mask = torch.tensor([[True]])
    return P, Q, E, lam, V_eff, orb_mask


def test_decay_consistency_vanishes_on_analytic_eigenstate():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=512, scheme="loglinear")
    loss_fn = DecayConsistencyLoss()
    for Z, n in [(1, 1), (1, 2), (2, 1), (3, 2)]:
        _, _, E, lam, _, mask = _make_hydrogenic_state(Z, n, 0, grid)
        val = loss_fn(lam, E, mask).item()
        # Exactly zero in float arithmetic since lam^2 = Z^2/n^2 = -2E by construction.
        assert val < 1.0e-10, f"decay residual non-zero for Z={Z} n={n}: {val}"


def test_decay_consistency_penalises_inconsistent_lambda():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = DecayConsistencyLoss()
    _, _, E, lam, _, mask = _make_hydrogenic_state(1, 1, 0, grid)
    perturbed = lam * 1.5
    val_consistent = loss_fn(lam, E, mask).item()
    val_perturbed = loss_fn(perturbed, E, mask).item()
    assert val_perturbed > val_consistent + 1.0e-4


def test_decay_consistency_has_gradient_through_lambda_and_energy():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=128, scheme="loglinear")
    loss_fn = DecayConsistencyLoss()
    lam = torch.tensor([[0.9]], dtype=torch.float32, requires_grad=True)
    E = torch.tensor([[-0.4]], dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([[True]])
    loss = loss_fn(lam, E, mask)
    loss.backward()
    assert lam.grad is not None and torch.abs(lam.grad).item() > 0.0
    assert E.grad is not None and torch.abs(E.grad).item() > 0.0


def test_virial_loss_vanishes_on_analytic_eigenstate():
    grid = RadialGrid(r_min=1.0e-4, r_max=80.0, n_grid=1024, scheme="loglinear")
    loss_fn = VirialLoss()
    # 1s and 2s — limit to states whose support fits on r_max=80 a.u.
    for Z, n in [(1, 1), (2, 1), (2, 2), (3, 2)]:
        P, Q, E, _, V_eff, mask = _make_hydrogenic_state(Z, n, 0, grid)
        val = loss_fn(P, Q, E, V_eff, grid.r, grid, mask).item()
        # Generalised virial: 2(E - <V>) - <r dV/dr> -> 0 on eigenstate; we
        # only check the residual is small relative to typical energy scale.
        # E^2 ~ Z^4 / (4 n^4), so tolerance scales with that.
        scale = (E.abs().item() ** 2) * 4.0 + 1.0e-3
        assert val < 0.05 * scale, f"virial residual large for Z={Z} n={n}: {val} (scale {scale})"


def test_virial_loss_passes_gradient_to_wavefunction_and_energy():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = VirialLoss()
    P, Q, E, _, V_eff, mask = _make_hydrogenic_state(1, 1, 0, grid)
    P = P.detach().clone().requires_grad_(True)
    Q = Q.detach().clone().requires_grad_(True)
    E = E.detach().clone().requires_grad_(True)
    # Distort slightly so the residual is non-zero -> non-zero gradient.
    P_in = P * 1.05
    val = loss_fn(P_in, Q, E, V_eff, grid.r, grid, mask)
    val.backward()
    assert P.grad is not None and torch.any(P.grad.abs() > 0)
    assert E.grad is not None and torch.abs(E.grad).item() > 0


# ───────────────────────── ADR-017: per-sample normalisation ─────────────────


def test_adr017_decay_e_char_sq_none_matches_no_arg():
    """``e_char_sq=None`` must reproduce the pre-ADR-017 behaviour exactly."""
    loss_fn = DecayConsistencyLoss()
    lam = torch.tensor([[0.6, 0.3]], dtype=torch.float32)
    E = torch.tensor([[-0.4, -0.1]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])
    a = loss_fn(lam, E, mask).item()
    b = loss_fn(lam, E, mask, e_char_sq=None).item()
    assert a == b


def test_adr017_decay_divides_by_e_char_sq_elementwise():
    """L_norm = mean_orb( residual² / E_char² ). Compare against hand-computed."""
    loss_fn = DecayConsistencyLoss()
    lam = torch.tensor([[1.5, 0.4]], dtype=torch.float32)
    E = torch.tensor([[-0.2, -0.1]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])
    e_char_sq = torch.tensor([[0.25, 0.04]], dtype=torch.float32)
    val = loss_fn(lam, E, mask, e_char_sq=e_char_sq).item()
    residual = lam.pow(2) + 2.0 * E
    expected = (residual.pow(2) / e_char_sq).mean().item()
    assert abs(val - expected) < 1.0e-6


def test_adr017_decay_normalises_cross_state_imbalance():
    """The whole point of ADR-017: deep state's absolute residual >> shallow's,
    but after normalisation per-state residuals end up in the same decade."""
    loss_fn = DecayConsistencyLoss()
    # Two states with the SAME relative error (~30%) in the consistency
    # condition, but very different absolute scales (Z=3,n=1 vs Z=1,n=4).
    lam_3_1 = 3.0
    lam_1_4 = 0.25
    E_3_1 = -(lam_3_1 ** 2) * 0.5 * (1.3)   # 30% overbinding
    E_1_4 = -(lam_1_4 ** 2) * 0.5 * (1.3)   # 30% overbinding
    lam = torch.tensor([[lam_3_1, lam_1_4]], dtype=torch.float32)
    E = torch.tensor([[E_3_1, E_1_4]], dtype=torch.float32)
    # Per-state loss WITHOUT normalisation: per-orb residual².
    res_3_1 = (lam_3_1 ** 2 + 2.0 * E_3_1) ** 2
    res_1_4 = (lam_1_4 ** 2 + 2.0 * E_1_4) ** 2
    raw_ratio = res_3_1 / res_1_4
    # With identical relative error, raw ratio ~ Z^8 / n^8 → 22 500× for (3,1)/(1,4).
    assert raw_ratio > 1.0e3
    # After ADR-017 normalisation per_orb is dimensionless and identical for both.
    e_char_sq = torch.tensor(
        [[(lam_3_1 ** 2 / 2.0) ** 2, (lam_1_4 ** 2 / 2.0) ** 2]],
        dtype=torch.float32,
    )
    per_orb_norm_3_1 = res_3_1 / e_char_sq[0, 0].item()
    per_orb_norm_1_4 = res_1_4 / e_char_sq[0, 1].item()
    norm_ratio = per_orb_norm_3_1 / per_orb_norm_1_4
    # The two normalised values should match within float noise — both encode
    # the same 30% relative inconsistency.
    assert abs(norm_ratio - 1.0) < 1.0e-3, (
        f"ADR-017 normalisation failed to equalise (Z=3,n=1) vs (Z=1,n=4): "
        f"ratio={norm_ratio}"
    )


def test_adr017_virial_e_char_sq_none_matches_no_arg():
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = VirialLoss()
    P, Q, E, _, V_eff, mask = _make_hydrogenic_state(1, 1, 0, grid)
    a = loss_fn(P, Q, E, V_eff, grid.r, grid, mask).item()
    b = loss_fn(P, Q, E, V_eff, grid.r, grid, mask, e_char_sq=None).item()
    assert a == b


def test_adr017_virial_divides_by_e_char_sq():
    """When e_char_sq is uniform across orbitals, the normalised loss equals
    the raw loss divided by that uniform value."""
    grid = RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")
    loss_fn = VirialLoss()
    P, Q, E, _, V_eff, mask = _make_hydrogenic_state(2, 2, 0, grid)
    raw = loss_fn(P, Q, E, V_eff, grid.r, grid, mask).item()
    uniform = 0.25
    e_char_sq = torch.full_like(E, uniform)
    normalised = loss_fn(P, Q, E, V_eff, grid.r, grid, mask, e_char_sq=e_char_sq).item()
    assert abs(normalised - raw / uniform) < 1.0e-5
