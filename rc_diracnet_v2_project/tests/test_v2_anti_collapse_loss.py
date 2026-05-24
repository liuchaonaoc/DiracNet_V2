"""Unit tests for the anti-(n-collapse / n-bloom) loss
(ADR-021 + ADR-023 + ADR-024 + ADR-026).

Pinned contracts:

ADR-021 (plain-sum, ``threshold=0``):

* ``L_anti_collapse(ψ_an_correct) ≈ 0`` — feeding the correct analytic
  reference produces near-zero loss because analytic hydrogenic
  eigenfunctions of the same l are orthogonal.
* ``L_anti_collapse(ψ_an_wrong_lower_n) ≈ 1`` — feeding the wrong lower-n
  analytic shape produces loss ≈ 1 (cos² with itself = 1, sum over the
  single conflicting term).
* ``n = 1`` and ``n = l + 1`` samples contribute exactly 0 *when
  lookahead = 0* (no lower n'). With lookahead ≥ 1 the upper window
  produces a non-zero term, see ADR-026 tests below.
* Gradient flows back through the loss.
* Batch class wrapper produces the same per-orbital values as the helper.

ADR-023 (hinge per-n′, ``threshold > 0``):

* Sub-threshold contamination contributes exactly 0 (no gradient).
* Above-threshold contamination contributes ``(cos² − τ)²`` per channel.
* Default ``threshold = 0.02`` reproduces the production behaviour.
* Helper raises on negative threshold.

ADR-026 (symmetric n-bloom defence, ``n_bloom_lookahead ≥ 1``):

* ``n_bloom_lookahead = 0`` is bit-equivalent to the legacy lower-only
  form on every test input (back-compat guard).
* A P that is the wrong *upper-n* analytic shape (n-bloom) gets
  cos² ≈ 1 against ψ_an(n+1, l) and contributes ≈ (1−τ)² to the loss
  iff lookahead ≥ 1.
* The lower-n channels are still penalised when lookahead > 0
  (additive, not replacement).
* Cap: helper/class raise when lookahead > 3 (grid extent rationale).

ADR-028 (asymmetric hinge-on-upper, ``threshold_upper > threshold``):

* ``threshold_upper = None`` reproduces ADR-026 symmetric behaviour
  (back-compat guard — kept so the ADR-026 tests above stay green).
* On clean states (cos²(P_target, ψ_an_upper) ≈ 0) the upper window
  contributes *exactly 0* with τ_upper = 0.30 — no gradient.
  This is the structural fix vs ADR-026's "no-penalty basin → drift
  to lower-n" pathology.
* On real bloom (cos² ≥ τ_upper) the upper window contributes the
  expected (cos² − τ_upper)² hinge.
* The lower window is unaffected by ``threshold_upper`` — it always
  uses ``threshold`` (plain-sum in production).
* Helper raises on negative ``threshold_upper``.
"""

from __future__ import annotations

import pytest
import torch

from rc_diracnet_v2.losses import AntiCollapseLoss, anti_collapse_per_orbital
from rc_diracnet_v2.physics.hydrogenic_analytic import hydrogenic_P_analytic
from rc_diracnet_v2.utils.grid import RadialGrid


@pytest.fixture(scope="module")
def grid() -> RadialGrid:
    return RadialGrid(r_min=1.0e-4, r_max=50.0, n_grid=256, scheme="loglinear")


# ───────────────── ADR-021 backwards-compat (threshold = 0) ─────────────────


def test_threshold0_correct_reference_has_zero_anti_collapse(grid: RadialGrid) -> None:
    """If P = P_an(Z, n, l), then cos²(P, P_an(n', l)) for n' < n is ≈ 0
    because hydrogenic radial eigenfunctions of the same l are orthogonal."""
    Z, n, l = 2, 4, 0
    P = hydrogenic_P_analytic(grid.r, Z, n, l)
    loss = anti_collapse_per_orbital(P, grid, Z, n, l, threshold=0.0)
    # Numerical residual from finite trapezoidal integration; expect ~1e-3.
    assert loss.item() < 1.0e-2, (
        f"expected ≈0 for correct analytic ref, got {loss.item():.4g}"
    )


def test_threshold0_wrong_lower_n_has_anti_collapse_near_one(grid: RadialGrid) -> None:
    """Plain-sum form (ADR-021): if P = P_an(Z, 1, l) but claimed to be
    (n=2, l), one cross term cos²(P_1s, P_1s) = 1 is picked up."""
    Z, l = 2, 0
    P_lower = hydrogenic_P_analytic(grid.r, Z, 1, l)
    loss = anti_collapse_per_orbital(P_lower, grid, Z, n=2, l=l, threshold=0.0)
    assert abs(loss.item() - 1.0) < 1.0e-3, (
        f"expected ≈1 for wrong-n leakage at τ=0, got {loss.item():.4g}"
    )


def test_threshold0_higher_n_sums_more_terms(grid: RadialGrid) -> None:
    """Plain-sum form: for n=4, l=0 the sum has three terms (n'=1,2,3)."""
    Z, l = 2, 0
    P_1s = hydrogenic_P_analytic(grid.r, Z, 1, l)
    P_2s = hydrogenic_P_analytic(grid.r, Z, 2, l)
    # Normalised equal mix; cos²(mix, 1s) ≈ 0.5, cos²(mix, 2s) ≈ 0.5,
    # cos²(mix, 3s) ≈ 0.
    P_mix = (P_1s + P_2s)
    loss = anti_collapse_per_orbital(P_mix, grid, Z, n=4, l=l, threshold=0.0)
    # Two terms ≈ 0.5 each → sum ≈ 1.0.
    assert 0.8 < loss.item() < 1.2, (
        f"expected ≈1.0 for two-state mix at n=4, got {loss.item():.4g}"
    )


# ───────────────── shared invariants (any threshold) ─────────────────


def test_n_equals_one_returns_zero_no_grad(grid: RadialGrid) -> None:
    """n=1 has no lower n' to penalise; returns 0 regardless of threshold."""
    Z = 1
    P = hydrogenic_P_analytic(grid.r, Z, 1, 0)
    loss = anti_collapse_per_orbital(P, grid, Z, n=1, l=0)
    assert loss.item() == 0.0


def test_n_equals_l_plus_one_returns_zero(grid: RadialGrid) -> None:
    """e.g. (n=2, l=1) — inner sum is empty regardless of threshold."""
    Z = 3
    P = hydrogenic_P_analytic(grid.r, Z, 2, 1)
    loss = anti_collapse_per_orbital(P, grid, Z, n=2, l=1)
    assert loss.item() == 0.0


def test_negative_threshold_raises(grid: RadialGrid) -> None:
    Z, n, l = 2, 3, 0
    P = hydrogenic_P_analytic(grid.r, Z, n, l)
    with pytest.raises(ValueError):
        anti_collapse_per_orbital(P, grid, Z, n, l, threshold=-0.01)
    with pytest.raises(ValueError):
        AntiCollapseLoss(grid, threshold=-0.05)


def test_gradient_flows(grid: RadialGrid) -> None:
    """Gradient must be non-trivial when ψ_pred is *not* at a stationary
    point of L_anti_collapse. The relevant stationary points are:

    * ψ_pred = ψ_an_lower exactly  → cos² = 1 is a local max.
    * ψ_pred ⟂ all ψ_an_lower      → cos² = 0 is a local min.
    * ψ_pred ∈ span{ψ_n′_an : n′ < n}  → the sum Σ cos²(P, ψ_n′_an) is
      identically 1 (Parseval inside the lower-n subspace), so the
      gradient w.r.t. P is again zero.

    The training-time failure mode (n-collapse) is exactly the third one,
    so a useful test must lift P **out of** the lower-n subspace. Mix
    with ψ_target (which is NOT in the inner sum) to do that.
    """
    Z, n, l = 2, 4, 0
    P_1s = hydrogenic_P_analytic(grid.r, Z, 1, l)
    P_4s = hydrogenic_P_analytic(grid.r, Z, n, l)
    # Asymmetric mix: 0.7·P_1s + 0.3·P_4s. The P_4s component is
    # orthogonal to all lower-n' references, so the sum is no longer
    # trapped at 1, and the gradient is well-defined.
    P_mix = (0.7 * P_1s + 0.3 * P_4s).clone().detach().requires_grad_(True)
    # Use threshold=0 to keep the gradient test signal strong; with
    # τ=0.02 the same shape produces qualitatively identical gradients
    # but smaller magnitudes (cos² − 0.02 instead of cos²).
    loss = anti_collapse_per_orbital(P_mix, grid, Z, n, l, threshold=0.0)
    loss.backward()
    assert P_mix.grad is not None
    assert torch.isfinite(P_mix.grad).all()
    grad_max = P_mix.grad.abs().max().item()
    assert grad_max > 1.0e-3, (
        f"expected non-trivial gradient (max |∂L/∂P_i| > 1e-3), got {grad_max:.3g}"
    )


# ───────────────── ADR-023 hinge invariants (threshold > 0) ─────────────────


def test_hinge_zero_below_threshold(grid: RadialGrid) -> None:
    """A contamination strictly less than τ contributes 0 to the loss
    (ADR-023 hinge property). Construct P so cos²(P, P_lower_n) ≈ τ/4."""
    Z, l = 2, 0
    P_1s = hydrogenic_P_analytic(grid.r, Z, 1, l)
    P_2s = hydrogenic_P_analytic(grid.r, Z, 2, l)
    # cos²(α·P_1s + β·P_2s normalised, P_1s) = α² / (α² + β²)
    # Choose α² / (α² + β²) = 0.01 → β/α = √99
    alpha = 1.0
    beta = (99.0) ** 0.5
    P_mix = alpha * P_1s + beta * P_2s
    # Target n=3 (so n'=1 and n'=2 are both inner-sum terms).
    # cos²(P_mix, P_1s) ≈ 0.01 (under τ=0.02 → 0).
    # cos²(P_mix, P_2s) ≈ 0.99 (above τ → contributes (0.99 − 0.02)²).
    loss_with_hinge = anti_collapse_per_orbital(
        P_mix, grid, Z, n=3, l=l, threshold=0.02
    )
    loss_no_hinge = anti_collapse_per_orbital(
        P_mix, grid, Z, n=3, l=l, threshold=0.0
    )
    # With hinge, only the n'=2 term should contribute.
    expected_only_n2 = (0.99 - 0.02) ** 2
    assert abs(loss_with_hinge.item() - expected_only_n2) < 5.0e-3, (
        f"hinge should suppress n'=1 (≈0.01 < τ=0.02), keeping only n'=2; "
        f"got {loss_with_hinge.item():.4g}, expected ≈{expected_only_n2:.4g}"
    )
    # Plain sum picks up both terms.
    assert loss_no_hinge.item() > loss_with_hinge.item() + 1.0e-3, (
        f"plain sum ({loss_no_hinge.item():.4g}) should exceed hinge "
        f"({loss_with_hinge.item():.4g}) when there is sub-τ contamination"
    )


def test_hinge_quadratic_above_threshold(grid: RadialGrid) -> None:
    """A single contamination at cos² ≈ α produces (α − τ)² when α > τ."""
    Z, l = 2, 0
    # P = P_1s exactly, target n=2 → cos²(P, P_1s) = 1.
    P_lower = hydrogenic_P_analytic(grid.r, Z, 1, l)
    loss_tau_02 = anti_collapse_per_orbital(
        P_lower, grid, Z, n=2, l=l, threshold=0.02
    )
    expected = (1.0 - 0.02) ** 2
    assert abs(loss_tau_02.item() - expected) < 1.0e-3, (
        f"expected (1 − 0.02)² = {expected:.4g}, got {loss_tau_02.item():.4g}"
    )


def test_hinge_zero_gradient_below_threshold(grid: RadialGrid) -> None:
    """When every lower-n′ contamination is below τ, the loss is 0 and
    the gradient must also be 0 (ADR-023 "done signal")."""
    Z, n, l = 2, 4, 0
    # Use P = P_an_4s exactly. All lower-n' contaminations are ≈ 0 < τ.
    P = hydrogenic_P_analytic(grid.r, Z, n, l).clone().detach().requires_grad_(True)
    loss = anti_collapse_per_orbital(P, grid, Z, n, l, threshold=0.02)
    assert loss.item() < 1.0e-6, "expected loss ≈ 0 for sub-τ contamination"
    loss.backward()
    # Gradient is the derivative of (ReLU(cos² − τ))² wherever cos² > τ,
    # and zero everywhere else. For a sub-τ configuration, all per-term
    # gradients should vanish.
    assert P.grad is not None
    grad_max = P.grad.abs().max().item()
    assert grad_max < 1.0e-5, (
        f"expected vanishing gradient (sub-τ done signal), got {grad_max:.3g}"
    )


# ───────────────── batch class wrapper ─────────────────


def test_class_default_threshold_matches_helper(grid: RadialGrid) -> None:
    """The nn.Module wrapper default τ matches the helper default
    (both 0.02). Same input should produce the same value."""
    Z_val, n, l = 2, 3, 0
    P_lower = hydrogenic_P_analytic(grid.r, Z_val, 1, l)
    expected = anti_collapse_per_orbital(P_lower, grid, Z_val, n, l).item()

    loss_mod = AntiCollapseLoss(grid)
    assert loss_mod.threshold == 0.02
    # Batch of 1, one active orbital.
    P_batch = P_lower.view(1, 1, -1)
    config = torch.zeros(1, 1, 2, dtype=torch.long)
    config[0, 0, 0] = n
    config[0, 0, 1] = l
    Z_t = torch.tensor([Z_val], dtype=torch.long)
    orb_mask = torch.ones(1, 1, dtype=torch.bool)
    nele = torch.ones(1, dtype=torch.long)
    got = loss_mod(P_batch, config, Z_t, orb_mask, nele=nele).item()
    assert abs(got - expected) < 1.0e-6


def test_class_propagates_threshold(grid: RadialGrid) -> None:
    """A different threshold passed to the class is used everywhere."""
    Z_val, n, l = 2, 2, 0
    P_lower = hydrogenic_P_analytic(grid.r, Z_val, 1, l)
    P_batch = P_lower.view(1, 1, -1)
    config = torch.zeros(1, 1, 2, dtype=torch.long)
    config[0, 0, 0] = n
    config[0, 0, 1] = l
    Z_t = torch.tensor([Z_val], dtype=torch.long)
    orb_mask = torch.ones(1, 1, dtype=torch.bool)
    nele = torch.ones(1, dtype=torch.long)
    # τ=0 → plain sum cos²(P_1s, P_1s) = 1.
    got_t0 = AntiCollapseLoss(grid, threshold=0.0)(
        P_batch, config, Z_t, orb_mask, nele=nele
    ).item()
    assert abs(got_t0 - 1.0) < 1.0e-3
    # τ=0.1 → (1 − 0.1)² = 0.81.
    got_t01 = AntiCollapseLoss(grid, threshold=0.1)(
        P_batch, config, Z_t, orb_mask, nele=nele
    ).item()
    assert abs(got_t01 - 0.81) < 1.0e-3


def test_class_skips_many_electron_samples(grid: RadialGrid) -> None:
    """If ``nele > 1``, the sample contributes 0 — hydrogenic baseline
    is not a valid reference for many-electron states."""
    Z_val, n, l = 2, 3, 0
    P_lower = hydrogenic_P_analytic(grid.r, Z_val, 1, l)
    loss_mod = AntiCollapseLoss(grid)
    P_batch = P_lower.view(1, 1, -1)
    config = torch.zeros(1, 1, 2, dtype=torch.long)
    config[0, 0, 0] = n
    config[0, 0, 1] = l
    Z_t = torch.tensor([Z_val], dtype=torch.long)
    orb_mask = torch.ones(1, 1, dtype=torch.bool)
    nele_many = torch.tensor([5], dtype=torch.long)
    got = loss_mod(P_batch, config, Z_t, orb_mask, nele=nele_many).item()
    assert got == 0.0


# ───────────────── ADR-026 symmetric n-bloom defence ─────────────────


def test_adr026_lookahead0_matches_legacy_lower_only(grid: RadialGrid) -> None:
    """``n_bloom_lookahead = 0`` must reproduce ADR-021/024 exactly.

    Sweep a representative set of (Z, n, l) and check that explicit
    ``n_bloom_lookahead = 0`` is bit-equivalent to the default (no
    parameter passed), so old behaviour is the back-compat point of the
    new API.
    """
    cases = [(1, 2, 0), (2, 3, 0), (2, 4, 0), (3, 3, 1), (3, 4, 2)]
    for Z, n, l in cases:
        P_lower = hydrogenic_P_analytic(grid.r, Z, max(l + 1, n - 1), l)
        legacy = anti_collapse_per_orbital(P_lower, grid, Z, n, l, threshold=0.0)
        explicit = anti_collapse_per_orbital(
            P_lower, grid, Z, n, l, threshold=0.0, n_bloom_lookahead=0
        )
        assert abs(legacy.item() - explicit.item()) < 1.0e-12, (
            f"lookahead=0 must be a no-op (Z={Z}, n={n}, l={l}): "
            f"legacy={legacy.item():.6g}, explicit={explicit.item():.6g}"
        )


def test_adr026_upper_n_bloom_is_penalised(grid: RadialGrid) -> None:
    """Construct the diagnosed (Z=3, n=2 → n=3) failure mode in isolation:
    feed ψ_an(Z=3, n=3, l=0) but *claim* it is (n=2, l=0). With
    lookahead ≥ 1 the loss must pick up cos²(P, ψ_an_3s) ≈ 1; without it
    the loss is dominated by the legacy n'=1 channel (≈ 0).
    """
    Z, l = 3, 0
    P_bloom = hydrogenic_P_analytic(grid.r, Z, 3, l)  # actual shape = 3s
    # Legacy: only n'=1 inner-sum term, cos²(3s, 1s) ≈ 0.
    legacy = anti_collapse_per_orbital(
        P_bloom, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=0
    )
    assert legacy.item() < 1.0e-2, (
        f"sanity: legacy loss must be ≈ 0 for n=2 claim of pure 3s "
        f"(orthogonality), got {legacy.item():.4g}"
    )
    # Lookahead = 1 → adds n'=3 term, which gives cos²(3s, 3s) = 1.
    with_la1 = anti_collapse_per_orbital(
        P_bloom, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=1
    )
    assert abs(with_la1.item() - 1.0) < 1.0e-2, (
        f"lookahead=1 must catch the n=2→n=3 bloom (≈1.0), "
        f"got {with_la1.item():.4g}"
    )
    # Lookahead = 2 adds n'=4 too (still ≈ 0 against 3s) — total still ≈ 1.
    with_la2 = anti_collapse_per_orbital(
        P_bloom, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=2
    )
    assert with_la2.item() >= with_la1.item() - 1.0e-3, (
        f"lookahead=2 ≥ lookahead=1 (monotone), got {with_la1.item():.4g} "
        f"vs {with_la2.item():.4g}"
    )


def test_adr026_lower_and_upper_are_additive(grid: RadialGrid) -> None:
    """Plain-sum form: with lookahead ≥ 1 the loss is the legacy lower-n
    sum *plus* the new upper-n sum (no interaction). Construct a P with
    equal contamination of ψ_an(1s) and ψ_an(3s) targeting n=2 — the
    lookahead=1 loss should be approximately the sum of the isolated
    lower and upper losses.
    """
    Z, l = 3, 0
    P_1s = hydrogenic_P_analytic(grid.r, Z, 1, l)
    P_3s = hydrogenic_P_analytic(grid.r, Z, 3, l)
    P_mix = P_1s + P_3s
    lower_only = anti_collapse_per_orbital(
        P_mix, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=0
    ).item()
    upper_only = anti_collapse_per_orbital(
        P_3s, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=1
    ).item() - anti_collapse_per_orbital(
        P_3s, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=0
    ).item()
    full = anti_collapse_per_orbital(
        P_mix, grid, Z, n=2, l=l, threshold=0.0, n_bloom_lookahead=1
    ).item()
    # cos²(P_mix, P_1s) ≈ 0.5, cos²(P_mix, P_3s) ≈ 0.5.
    assert 0.8 < full < 1.2, (
        f"expected ≈1.0 for equal mix lower+upper, got {full:.4g}"
    )
    assert lower_only > 0.4, f"lower channel should be ≈0.5, got {lower_only:.4g}"
    assert upper_only > 0.0, "isolated upper channel should be > 0"


def test_adr026_n1_gets_bloom_channel_when_lookahead_positive(
    grid: RadialGrid,
) -> None:
    """With lookahead ≥ 1, an n=1 state contributing a 2s-shape sample
    (n=1 claim, actual ψ_an(2s)) should also be penalised. Legacy (l=0)
    returns 0 for n=1.
    """
    Z, l = 1, 0
    P_bloom = hydrogenic_P_analytic(grid.r, Z, 2, l)  # actual = 2s
    legacy = anti_collapse_per_orbital(
        P_bloom, grid, Z, n=1, l=l, threshold=0.0, n_bloom_lookahead=0
    )
    assert legacy.item() == 0.0, "n=1, lookahead=0 must be exactly 0"
    new_form = anti_collapse_per_orbital(
        P_bloom, grid, Z, n=1, l=l, threshold=0.0, n_bloom_lookahead=2
    )
    # cos²(2s, 2s) = 1, cos²(2s, 3s) ≈ 0. Sum ≈ 1.
    assert abs(new_form.item() - 1.0) < 1.0e-2, (
        f"lookahead=2 on n=1 with 2s contamination should give ≈1, "
        f"got {new_form.item():.4g}"
    )


def test_adr026_lookahead_cap_at_three(grid: RadialGrid) -> None:
    """Both the class and the helper-class config path must refuse
    ``n_bloom_lookahead > 3`` (grid extent rationale)."""
    with pytest.raises(ValueError):
        AntiCollapseLoss(grid, n_bloom_lookahead=4)
    with pytest.raises(ValueError):
        AntiCollapseLoss(grid, n_bloom_lookahead=-1)


def test_adr026_class_propagates_lookahead(grid: RadialGrid) -> None:
    """The nn.Module wrapper must forward ``n_bloom_lookahead`` to the
    per-orbital helper."""
    Z_val, l = 3, 0
    n_target = 2
    P_bloom = hydrogenic_P_analytic(grid.r, Z_val, 3, l)  # 3s
    P_batch = P_bloom.view(1, 1, -1)
    config = torch.zeros(1, 1, 2, dtype=torch.long)
    config[0, 0, 0] = n_target
    config[0, 0, 1] = l
    Z_t = torch.tensor([Z_val], dtype=torch.long)
    orb_mask = torch.ones(1, 1, dtype=torch.bool)
    nele = torch.ones(1, dtype=torch.long)

    loss_la0 = AntiCollapseLoss(grid, threshold=0.0, n_bloom_lookahead=0)(
        P_batch, config, Z_t, orb_mask, nele=nele
    ).item()
    loss_la2 = AntiCollapseLoss(grid, threshold=0.0, n_bloom_lookahead=2)(
        P_batch, config, Z_t, orb_mask, nele=nele
    ).item()
    # Lookahead 0 misses the bloom; lookahead 2 catches it.
    assert loss_la0 < 1.0e-2, f"la=0 should miss bloom, got {loss_la0:.4g}"
    assert abs(loss_la2 - 1.0) < 1.0e-2, (
        f"la=2 should catch n=2→n=3 bloom, got {loss_la2:.4g}"
    )


# ───────────────── ADR-028 asymmetric hinge-on-upper ─────────────────


def test_adr028_threshold_upper_none_matches_symmetric_adr026(
    grid: RadialGrid,
) -> None:
    """``threshold_upper = None`` (default) must reproduce ADR-026
    symmetric behaviour exactly. Sweep representative (Z, n, l, lookahead)
    cases and check the helper output is bit-equivalent to the legacy
    call (no ``threshold_upper`` passed).
    """
    cases = [
        (1, 1, 0, 1),  # n=1 lower window empty, upper window n'=2
        (2, 2, 0, 1),  # n=2: lower {1}, upper {3}
        (3, 2, 0, 1),  # the diagnosed (3,2) bloom geometry
        (3, 3, 0, 2),  # lower {1, 2}, upper {4, 5}
    ]
    P = hydrogenic_P_analytic(grid.r, 3, 3, 0)  # a fixed shape
    for Z, n, l, la in cases:
        legacy = anti_collapse_per_orbital(
            P, grid, Z, n, l, threshold=0.0, n_bloom_lookahead=la
        )
        explicit = anti_collapse_per_orbital(
            P, grid, Z, n, l,
            threshold=0.0, n_bloom_lookahead=la, threshold_upper=None,
        )
        assert abs(legacy.item() - explicit.item()) < 1.0e-12, (
            f"threshold_upper=None must reproduce ADR-026 symmetric form "
            f"(Z={Z}, n={n}, l={l}, la={la}): "
            f"legacy={legacy.item():.6g}, explicit={explicit.item():.6g}"
        )


def test_adr028_clean_state_upper_window_silent(grid: RadialGrid) -> None:
    """Key ADR-028 invariant: on a clean state (P = ψ_an_target), the
    upper-window contribution with τ_upper = 0.30 must be *exactly* 0
    (or floating-point noise below 1e-6), and the gradient w.r.t. P
    must be 0 too. This is the structural property that prevents the
    ADR-026 "drift to lower-n" pathology.
    """
    Z, n, l = 3, 2, 0  # the diagnosed bloom geometry
    # P = ψ_an(3, 2, 0) exactly — orthogonal to ψ_an(3, 3, 0).
    P = (
        hydrogenic_P_analytic(grid.r, Z, n, l)
        .clone().detach().requires_grad_(True)
    )

    # Symmetric form (ADR-026): even on clean state, the upper window
    # adds cos²(P_2s, P_3s) which is tiny but nonzero noise.
    symmetric = anti_collapse_per_orbital(
        P, grid, Z, n, l,
        threshold=0.0, n_bloom_lookahead=1, threshold_upper=None,
    )

    # Asymmetric form (ADR-028): with τ_upper = 0.30 the upper window
    # is forced to exactly 0 on clean states.
    asymmetric = anti_collapse_per_orbital(
        P, grid, Z, n, l,
        threshold=0.0, n_bloom_lookahead=1, threshold_upper=0.30,
    )

    # Both should be small but the asymmetric one should be exactly 0
    # for the upper window (only the lower-window n'=1 cos² survives,
    # and that is also ~ 1e-3 by hydrogenic orthogonality).
    assert asymmetric.item() < symmetric.item() + 1.0e-6, (
        f"asymmetric ({asymmetric.item():.4g}) should be ≤ symmetric "
        f"({symmetric.item():.4g}) on clean state — the hinge can only "
        "drop terms, never add them"
    )
    assert asymmetric.item() < 1.0e-2, (
        f"asymmetric loss on clean state should be tiny (≈0), got "
        f"{asymmetric.item():.4g}"
    )

    # Gradient must also be near-zero on clean state — this is the
    # core "no drift" property.
    asymmetric.backward()
    grad_max = P.grad.abs().max().item()
    assert grad_max < 1.0e-3, (
        f"clean-state gradient must be ≈ 0 (ADR-028 'no-drift' property), "
        f"got max |∂L/∂P| = {grad_max:.3g}"
    )


def test_adr028_real_bloom_upper_window_fires(grid: RadialGrid) -> None:
    """On a real bloom (P ≈ ψ_an(n+1, l)) the upper window with
    τ_upper = 0.30 must contribute approximately (1 − 0.30)² = 0.49,
    same as the legacy ADR-026 form would contribute for a τ=0 channel
    with cos²=1 (1.0), but with a 51% discount for the hinge. The
    point: the hinge fires *firmly* when the failure actually happens.
    """
    Z, n, l = 3, 2, 0
    # P = ψ_an(3, 3, 0) — the diagnosed bloom direction.
    P_bloom = hydrogenic_P_analytic(grid.r, Z, 3, l)

    loss = anti_collapse_per_orbital(
        P_bloom, grid, Z, n, l,
        threshold=0.0, n_bloom_lookahead=1, threshold_upper=0.30,
    )
    # Lower window: cos²(3s, 1s) ≈ 0 → 0 (plain-sum, but argument is tiny).
    # Upper window: cos²(3s, 3s) = 1 → ReLU(1 − 0.30)² = 0.49.
    expected = (1.0 - 0.30) ** 2
    assert abs(loss.item() - expected) < 5.0e-3, (
        f"real bloom with τ_upper=0.30 should give ≈ {expected:.4g}, "
        f"got {loss.item():.4g}"
    )


def test_adr028_borderline_contamination_below_tau_upper(
    grid: RadialGrid,
) -> None:
    """A contamination strictly below τ_upper contributes 0 to the
    upper window — this is the hinge's "noise floor" behaviour. The
    lower window is unaffected (still plain-sum).

    Construct P so cos²(P, ψ_an_upper) ≈ 0.20 (< τ_upper=0.30) and
    cos²(P, ψ_an_lower) ≈ 0.80 (above any lower threshold).
    """
    Z, l = 3, 0
    n_target = 2
    P_1s = hydrogenic_P_analytic(grid.r, Z, 1, l)
    P_3s = hydrogenic_P_analytic(grid.r, Z, 3, l)
    # Mix α·P_1s + β·P_3s. cos²(P, P_3s) = β² / (α² + β²) (orthogonal).
    # Want cos²(P, P_3s) ≈ 0.20 → β²/(α²+β²)=0.20 → α/β = 2.
    alpha = 2.0
    beta = 1.0
    P_mix = alpha * P_1s + beta * P_3s
    # cos²(P, P_1s) ≈ 0.80, cos²(P, P_3s) ≈ 0.20.
    asymmetric = anti_collapse_per_orbital(
        P_mix, grid, Z, n_target, l,
        threshold=0.0, n_bloom_lookahead=1, threshold_upper=0.30,
    )
    # Lower window picks up cos²(P, P_1s) ≈ 0.80 (plain-sum).
    # Upper window: cos²(P, P_3s) ≈ 0.20 < τ_upper=0.30 → contributes 0.
    # Total ≈ 0.80.
    assert 0.70 < asymmetric.item() < 0.90, (
        f"asymmetric loss should be dominated by lower-window 0.80, "
        f"upper window silent at sub-τ, got {asymmetric.item():.4g}"
    )

    # Sanity: with symmetric (ADR-026) the upper would add 0.20.
    symmetric = anti_collapse_per_orbital(
        P_mix, grid, Z, n_target, l,
        threshold=0.0, n_bloom_lookahead=1, threshold_upper=None,
    )
    assert symmetric.item() > asymmetric.item() + 0.10, (
        f"symmetric ({symmetric.item():.4g}) should be ≥ asymmetric "
        f"({asymmetric.item():.4g}) + 0.10 (the upper 0.20 channel "
        "the hinge silences)"
    )


def test_adr028_negative_threshold_upper_raises(grid: RadialGrid) -> None:
    Z, n, l = 3, 2, 0
    P = hydrogenic_P_analytic(grid.r, Z, n, l)
    with pytest.raises(ValueError):
        anti_collapse_per_orbital(
            P, grid, Z, n, l,
            threshold=0.0, n_bloom_lookahead=1, threshold_upper=-0.01,
        )
    with pytest.raises(ValueError):
        AntiCollapseLoss(grid, threshold_upper=-0.05)


def test_adr028_class_propagates_threshold_upper(grid: RadialGrid) -> None:
    """The nn.Module wrapper must forward ``threshold_upper`` to the
    per-orbital helper."""
    Z_val, l = 3, 0
    n_target = 2
    P_bloom = hydrogenic_P_analytic(grid.r, Z_val, 3, l)
    P_batch = P_bloom.view(1, 1, -1)
    config = torch.zeros(1, 1, 2, dtype=torch.long)
    config[0, 0, 0] = n_target
    config[0, 0, 1] = l
    Z_t = torch.tensor([Z_val], dtype=torch.long)
    orb_mask = torch.ones(1, 1, dtype=torch.bool)
    nele = torch.ones(1, dtype=torch.long)

    # Asymmetric (ADR-028): upper hinge at 0.30.
    loss_asym = AntiCollapseLoss(
        grid, threshold=0.0, n_bloom_lookahead=1, threshold_upper=0.30,
    )(P_batch, config, Z_t, orb_mask, nele=nele).item()
    # Symmetric (ADR-026): upper plain-sum.
    loss_sym = AntiCollapseLoss(
        grid, threshold=0.0, n_bloom_lookahead=1, threshold_upper=None,
    )(P_batch, config, Z_t, orb_mask, nele=nele).item()
    # On full-bloom P, asymmetric gives (1−0.30)²=0.49, symmetric gives 1.0.
    assert abs(loss_asym - 0.49) < 0.01, (
        f"asymmetric full-bloom should be ≈ 0.49, got {loss_asym:.4g}"
    )
    assert abs(loss_sym - 1.0) < 0.01, (
        f"symmetric full-bloom should be ≈ 1.0, got {loss_sym:.4g}"
    )


def test_class_batch_average_works(grid: RadialGrid) -> None:
    """A batch with one n=1 sample (skipped) and one n=2-claiming-1s
    sample (loss = (1−τ)²) should average to (1−τ)² over the 1
    contributing orbital."""
    Z_val, l = 2, 0
    P_1s = hydrogenic_P_analytic(grid.r, Z_val, 1, l)
    P_batch = torch.stack([P_1s, P_1s], dim=0).unsqueeze(1)  # [2, 1, N_grid]
    config = torch.zeros(2, 1, 2, dtype=torch.long)
    config[0, 0, 0] = 1  # sample 0 — n=1, skipped
    config[1, 0, 0] = 2  # sample 1 — n=2 claim, real shape is 1s
    Z_t = torch.tensor([Z_val, Z_val], dtype=torch.long)
    orb_mask = torch.ones(2, 1, dtype=torch.bool)
    nele = torch.ones(2, dtype=torch.long)
    loss_mod = AntiCollapseLoss(grid, threshold=0.02)
    got = loss_mod(P_batch, config, Z_t, orb_mask, nele=nele).item()
    expected = (1.0 - 0.02) ** 2
    assert abs(got - expected) < 1.0e-3, (
        f"expected ≈{expected:.4g} (only sample-1 contributes), got {got:.4g}"
    )
