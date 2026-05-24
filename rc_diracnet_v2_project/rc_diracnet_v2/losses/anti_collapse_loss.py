"""Anti-(n-collapse/bloom) loss for hydrogenic single-electron data
(ADR-021, ADR-023, ADR-024, ADR-026, ADR-028).

Motivation
----------
Diagnostic `scripts/v2_diagnose_shape_loss.py` revealed that after 1000 epoch
the model systematically returns ψ_pred(n) whose dominant shape component
is the analytic ψ_an(n-1, l) (or even n-2). E.g. for (Z=2, n=4):
``cos²(ψ_pred, ψ_an_4s) = 0.015`` while ``cos²(ψ_pred, ψ_an_3s) = 0.826``.

The pre-existing `L_shape = 1 − cos²(ψ_pred, ψ_an(n,l))` only rewards
similarity to the *correct* reference; it cannot strongly suppress
overlap with a *different* analytic eigenstate. The Rayleigh quotient
trivially prefers any eigenstate of *H* with more negative energy, so
ψ_pred is happy to collapse onto a lower-n shape.

In the single-electron (hydrogenic) regime the ordinary cross-orbital
`L_ortho = ‖S − I‖²` is trivially zero because each sample has exactly
one active orbital, so it cannot supply the "stay away from lower n"
gradient either.

Anti-collapse loss (hinge-quadratic per lower-n; ADR-023 supersedes ADR-021)
---------------------------------------------------------------------------
Add an explicit Gram-Schmidt-like penalty with a **per-n′ hinge**:

    L_anti_collapse = (1 / N_active) · Σ_orb Σ_{n' = l+1}^{n−1}
                      ReLU(cos²(P_pred, P_an(n', l)) − τ)²

where ``τ`` (= ``threshold``, default 0.02) is the per-lower-n tolerance.
Setting ``τ = 0`` recovers the plain ``Σ cos²`` form of ADR-021.

The inner sum is taken over **analytically lower** principal quantum
numbers (same l). For l=0, n=1 this sum is empty (no lower n'), and
similarly for n = l + 1. For (n=2, l=0) the only term is n'=1
(suppress 1s leakage). For (n=4, l=0) the sum has three terms
(suppress 1s, 2s and 3s leakage).

Why per-n′ hinge instead of plain Σ cos² (ADR-023)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original plain-sum form (ADR-021) lets the model satisfy the
constraint *on average*: it can put 5% of leakage into each of three
lower-n shapes and still have Σ cos² ≈ 0.15 — small overall, but the
*energy impact* is the sum 5%·|E_n − E_{n′}|, which is exactly the linear
combination the model exploits. The 1000-epoch run after ADR-021 confirmed
this: per-(Z=2, n=2) error of −3139 meV matches a single 1s contamination
of α ≈ 7.7%  (E_pred = (1−α)·E_n + α·E_{n−1}), and the
``avg_anti_collapse ≈ 0.49`` at convergence corresponds to several
per-cent contamination spread across all lower-n′ together.

The hinge formulation puts a **per-component cap**: a single lower-n′
contamination of 1.5% contributes 0 to L (within tolerance), whereas a
single lower-n′ contamination of 5% contributes (0.05 − 0.02)² = 9·10⁻⁴
which is then weighted heavily. This converts the soft, averageable
constraint into a hard ceiling on each individual contamination channel,
which is what the Z=2 n=2 error mechanism actually needs.

Properties
----------
* Per-sample / per-orbital — no batch-level ortho coupling required, so
  it works on single-electron data where `L_ortho` is structurally 0.
* Cheap — analytic reference ``P_an(n', l)`` is computed deterministically
  from (Z, n', l) and the radial grid.
* Bounded — every term is in [0, (1 − τ)²]; the sum over n' < n is
  bounded by (n − l − 1)·(1 − τ)². With τ = 0.02 this is ≈ n − l − 1.
* Direct attack on the failure mode — gradients flow specifically through
  the overlap with the wrong-n reference, exactly the quantity the
  diagnostic showed to be large.
* Hinge boundary at τ — once a particular lower-n′ contamination drops
  below τ, that channel contributes zero gradient. This stops the loss
  from competing with the other losses (PDE, shape) once the threshold
  is met, and gives the optimiser a clear "done" signal.

Relationship to other losses
----------------------------
* `L_shape` continues to reward ``cos²(P_pred, P_an(n,l)) → 1``.
* Together: L_shape and L_anti_collapse jointly enforce that ψ_pred is
  *the* (n,l) eigenfunction, not just *some* eigenstate of H.
* No interaction with stage-2 NIST calibration (kept in stage-1 physics
  losses only).

Symmetric n-bloom defence (ADR-026, supersedes the lower-only window)
---------------------------------------------------------------------
After ADR-017 (per-sample normalisation, slice 2) the 1000-epoch run
exhibited a new failure mode at (Z=3, n=2): pred E flipped from
−1.31 eV (overbinding) to **+6.51 eV underbinding** because the
predicted P contained substantial overlap with the *upper* state ψ_an(3, l)
(``cos² ≈ 0.5+``), while envelope λ stayed close to the ref of n=2.
The original loss only sums over ``n′ < n``, so this "n-bloom" was
invisible to the constraint — the model exploited it once ADR-017
restored its gradient budget.

ADR-026 closes the asymmetry: the inner sum is widened to
``n′ ∈ [l+1, n + n_bloom_lookahead]`` minus ``{n}``, i.e. it now also penalises
``cos²(P_pred, ψ_an(n+1, l)), …, cos²(P_pred, ψ_an(n+n_bloom_lookahead, l))``.

ADR-026 was **Rejected** after a 1000-epoch run because using the same
plain-sum form (τ_upper = τ_lower = 0) for the upper window caused a
"no-penalty basin → drift to lower-n" pathology: even clean states
(P ≈ ψ_an_target) received a small upper-window gradient that pushed
P away from ψ_an(n+1, l), and the lowest-energy direction orthogonal
to that was ψ_an(n−1, l) → bloom flipped to deeper collapse. See the
ADR-026 Rejection rationale.

ADR-028 reopens the bloom defence with an **asymmetric hinge**: the
upper window uses ``threshold_upper > 0`` (default 0.30) while the
lower window keeps ``threshold = 0`` (plain-sum, ADR-024). With
τ_upper = 0.30:

* Clean state: ``cos²(P_target, ψ_an(n+1, l)) ≈ 0`` (orthogonal
  hydrogenic eigenstates), so ``ReLU(0 − 0.30) = 0`` — exact zero
  gradient. P is *not* pushed away from anywhere.
* Real bloom (e.g. ADR-017 (Z=3, n=2) sat at cos² ≈ 0.5+):
  ``ReLU(0.5 − 0.30)² = 0.04`` per channel × weight 5 ≈ 0.2 gradient
  signal — enough to push P off the bloom basin without dominating.
* This is the *opposite* of ADR-023 (which used a hinge on the
  *lower* window and failed): bloom requires high cos², so a hinge
  there filters noise; collapse can sit at small cos² spread across
  many lower-n′ (Parseval averageability), so the lower window must
  stay plain-sum to enforce per-channel ceilings.

`n_bloom_lookahead` and `threshold_upper` both default to "off"
(0 and None respectively) in the helper signature so legacy callers
are unaffected. The production default in `configs/default.yaml`
under ADR-028 is `n_bloom_lookahead: 1, threshold_upper: 0.30`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..physics.hydrogenic_analytic import hydrogenic_P_analytic


_MAX_N_BLOOM_LOOKAHEAD = 3
"""Hard cap on `n_bloom_lookahead`.

The analytic hydrogenic radial function ``hydrogenic_P_analytic`` grows in
extent like ``n²/Z``. On the default ``r_max = 50 a.u.`` grid, only
``n ≤ 6`` (for Z ≥ 1) is reliably resolved end-to-end. With the production
``n_max = 4`` and ``Z_max = 3`` in the hydrogenic dataset, ``n + 3 = 7`` is
already at the edge; cap at 3 to keep ADR-026 safe across future small
dataset extensions. Anything above 3 should re-evaluate the grid first.
"""


def anti_collapse_per_orbital(
    P: Tensor,
    grid,
    Z: int,
    n: int,
    l: int,
    threshold: float = 0.02,
    eps: float = 1.0e-30,
    n_bloom_lookahead: int = 0,
    threshold_upper: float | None = None,
) -> Tensor:
    r"""Return Σ_{n' ∈ [l+1, n+lookahead] \ {n}} ReLU(cos²(P, P_an(Z, n', l)) − τ_n')².

    With ``threshold = 0`` this reduces to the plain ``Σ cos²`` form
    (ADR-021). For the production default ``threshold = 0.02`` each
    contamination contributes 0 below 2% and a quadratic penalty above
    (ADR-023; reverted to ``threshold = 0`` in ADR-024).

    ``n_bloom_lookahead`` controls the upper window (ADR-026):

    - ``0`` — legacy behaviour: only ``n' ∈ [l+1, n-1]`` (lower-n only).
      Reproduces ADR-021 / ADR-024 form.
    - ``≥ 1`` — symmetric form: additionally penalises
      ``n' ∈ [n+1, n + n_bloom_lookahead]`` (upper-n / n-bloom).
      The case ``n' = n`` itself is always excluded (that overlap is
      what ``L_shape`` rewards).

    ``threshold_upper`` controls the per-channel threshold on the upper
    bloom window separately from the lower window (ADR-028):

    - ``None`` (default) — upper window uses the same ``threshold`` as
      the lower window. This is the symmetric ADR-026 form (Rejected as
      a production default). Kept as a default so old callers and unit
      tests for the symmetric form are unaffected.
    - ``≥ 0`` — upper window uses its own hinge threshold ``τ_upper``.
      The production setting (ADR-028) is ``threshold = 0`` (lower
      plain-sum) plus ``threshold_upper = 0.30`` (upper hinge): clean
      states (cos²(P_target, ψ_an(n+1, l)) ≈ 0) contribute 0 to the
      upper window, while real bloom (cos² ≥ 0.30) is penalised
      quadratically. This breaks the ADR-026 failure mode in which
      the upper window's continuous gradient on clean states drifted
      P toward lower-n.

    Returns a scalar tensor with value 0 (no graph) when the resulting
    set of valid ``n'`` is empty (e.g. ``n = 1, lookahead = 0``).

    Parameters
    ----------
    P : Tensor[N_grid]
        The predicted (large-component) radial function for a single orbital.
    grid : RadialGrid
        Provides ``grid.r`` and ``grid.integrate``.
    Z, n, l : int
        Quantum numbers labelling the *target* state of P.
    threshold : float, default 0.02
        Per-channel tolerance ``τ`` for the **lower** window (and for
        the upper window too if ``threshold_upper is None``).
    n_bloom_lookahead : int, default 0
        Upper-window width. ``0`` keeps the lower-only legacy form;
        positive values enable anti-bloom (ADR-026/ADR-028). Capped at
        ``_MAX_N_BLOOM_LOOKAHEAD`` (= 3) by ``AntiCollapseLoss`` config.
    threshold_upper : float or None, default None
        Per-channel tolerance for the **upper** bloom window. ``None``
        means "use ``threshold``" (symmetric ADR-026 form). Setting a
        value ``> threshold`` enables the asymmetric ADR-028 form where
        the upper window only fires on real bloom.
    """
    if threshold < 0.0:
        raise ValueError(f"anti_collapse threshold must be ≥ 0, got {threshold}")
    if n_bloom_lookahead < 0:
        raise ValueError(
            f"anti_collapse n_bloom_lookahead must be ≥ 0, got {n_bloom_lookahead}"
        )
    # Default upper threshold = lower threshold (back-compat for ADR-026 tests).
    tau_lower = float(threshold)
    tau_upper = float(threshold) if threshold_upper is None else float(threshold_upper)
    if tau_upper < 0.0:
        raise ValueError(
            f"anti_collapse threshold_upper must be ≥ 0, got {tau_upper}"
        )

    # Build the active n' set: lower window [l+1, n-1] plus the upper bloom
    # window [n+1, n+lookahead], excluding n itself.
    nprime_lower = list(range(l + 1, n))
    nprime_upper = list(range(n + 1, n + int(n_bloom_lookahead) + 1))
    nprimes_with_tau: list[tuple[int, float]] = (
        [(np_, tau_lower) for np_ in nprime_lower]
        + [(np_, tau_upper) for np_ in nprime_upper]
    )
    if not nprimes_with_tau:
        return P.sum() * 0.0

    p_norm = grid.integrate(P * P, dim=-1).clamp_min(eps).sqrt()
    terms: list[Tensor] = []
    for nprime, tau in nprimes_with_tau:
        Pref = hydrogenic_P_analytic(grid.r.to(P.device).to(P.dtype), Z, nprime, l)
        ref_norm = grid.integrate(Pref * Pref, dim=-1).clamp_min(eps).sqrt()
        num = grid.integrate(P * Pref, dim=-1)
        cos = num / (p_norm * ref_norm).clamp_min(eps)
        cos2 = cos.pow(2)
        if tau > 0.0:
            # Hinge-quadratic: penalise only the part above τ.
            terms.append(F.relu(cos2 - tau).pow(2))
        else:
            terms.append(cos2)
    if not terms:
        return P.sum() * 0.0
    return torch.stack(terms).sum()


class AntiCollapseLoss(nn.Module):
    """Anti-(n-collapse/bloom) loss for a batch of single-electron hydrogenic data.

    The loss is averaged over the active (one-electron) orbitals that have
    *at least one* n' in the active window (see ``anti_collapse_per_orbital``).
    With ``n_bloom_lookahead = 0`` only ``n > l + 1`` orbitals contribute;
    with ``n_bloom_lookahead ≥ 1`` even ``n = 1`` contributes (it sees
    n' = 2, …, 1 + lookahead as bloom channels).

    Parameters
    ----------
    grid : RadialGrid
        Radial grid for integration.
    threshold : float, default 0.02
        Per-channel tolerance. See ``anti_collapse_per_orbital``.
    n_bloom_lookahead : int, default 0
        Upper window width. ``0`` reproduces ADR-021/024 (lower-n only).
        Production default in ``configs/default.yaml`` is 2 (symmetric
        ADR-026 form). Hard capped at ``_MAX_N_BLOOM_LOOKAHEAD`` (= 3).
    """

    def __init__(
        self,
        grid,
        threshold: float = 0.02,
        n_bloom_lookahead: int = 0,
        threshold_upper: float | None = None,
    ) -> None:
        super().__init__()
        if threshold < 0.0:
            raise ValueError(f"anti_collapse threshold must be ≥ 0, got {threshold}")
        if n_bloom_lookahead < 0:
            raise ValueError(
                f"anti_collapse n_bloom_lookahead must be ≥ 0, got {n_bloom_lookahead}"
            )
        if n_bloom_lookahead > _MAX_N_BLOOM_LOOKAHEAD:
            raise ValueError(
                f"anti_collapse n_bloom_lookahead must be ≤ {_MAX_N_BLOOM_LOOKAHEAD} "
                f"(see grid-extent rationale in the module docstring), "
                f"got {n_bloom_lookahead}"
            )
        if threshold_upper is not None and float(threshold_upper) < 0.0:
            raise ValueError(
                f"anti_collapse threshold_upper must be ≥ 0 or None, "
                f"got {threshold_upper}"
            )
        self.grid = grid
        self.threshold = float(threshold)
        self.n_bloom_lookahead = int(n_bloom_lookahead)
        self.threshold_upper = (
            None if threshold_upper is None else float(threshold_upper)
        )

    def forward(
        self,
        P: Tensor,
        config_shells: Tensor,
        Z: Tensor,
        orb_mask: Tensor,
        nele: Tensor | None = None,
    ) -> Tensor:
        """Compute the anti-(collapse + bloom) loss.

        Parameters
        ----------
        P : Tensor[B, N_orb, N_grid]
        config_shells : Tensor[B, N_orb_cfg, ≥2] — column 0 is n, column 1 is l.
        Z : Tensor[B] — atomic number per sample.
        orb_mask : BoolTensor[B, N_orb]
        nele : Tensor[B] or None — number of electrons. Used to skip
            many-electron samples (this loss assumes hydrogenic / single-
            electron behaviour). If None, every active orbital is included.
        """
        device = P.device
        B = P.shape[0]
        N_orb_loss = min(P.shape[1], config_shells.shape[1])
        config = config_shells.to(device)
        Z_d = Z.to(device)
        if nele is not None:
            nele_mask = nele.to(device).view(-1, 1) <= 1  # [B, 1] bool
        else:
            nele_mask = torch.ones((B, 1), dtype=torch.bool, device=device)

        losses: list[Tensor] = []
        for b in range(B):
            if not bool(nele_mask[b, 0].item()):
                continue
            z_b = int(Z_d[b].item())
            for o in range(N_orb_loss):
                if not bool(orb_mask[b, o].item()):
                    continue
                n = int(config[b, o, 0].item())
                l = int(config[b, o, 1].item())
                if n <= 0:
                    continue
                # Skip only when *both* lower and upper windows are empty.
                # With lookahead ≥ 1, even n = 1 has a non-empty upper window.
                has_lower = n > l + 1
                has_upper = self.n_bloom_lookahead > 0
                if not (has_lower or has_upper):
                    continue
                losses.append(
                    anti_collapse_per_orbital(
                        P[b, o],
                        self.grid,
                        z_b,
                        n,
                        l,
                        threshold=self.threshold,
                        n_bloom_lookahead=self.n_bloom_lookahead,
                        threshold_upper=self.threshold_upper,
                    )
                )

        if not losses:
            return P.sum() * 0.0
        return torch.stack(losses).mean()
