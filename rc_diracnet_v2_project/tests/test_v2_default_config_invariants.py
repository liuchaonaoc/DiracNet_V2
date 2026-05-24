"""Invariants pinned on ``configs/default.yaml`` itself.

These tests do **not** train anything — they parse the YAML and assert that
specific architectural decisions recorded in ADRs cannot regress silently
through a config edit.

Currently pinned:

* ADR-020 — ``stage1.ortho.use_full_lowdin_after_epoch`` must be effectively
  disabled (≥ 1_000_000). The ``eigh``-based full Löwdin path blew up at
  epoch 201 in the slice-1 patch run (`avg_total = nan` from then on); the
  underlying mathematical reason is that ``eigh`` backward is undefined when
  the overlap matrix has near-duplicate eigenvalues, and the mean ``L_ortho``
  is not a valid proxy for the worst-case ``λ_min(S)``.
* ADR-020 corollary — ``stage1.weights.ortho`` must be ≥ 50.0 to compensate
  for the missing hard projection (``L_ortho`` is now the only mechanism
  enforcing cross-orbital orthogonality during stage 1).
* ADR-019 — the three deprecated local shape supervision losses
  (``node_pos``/``node_cross``/``sign``) must have weight 0.
* ADR-026 / ADR-028 (both Rejected) — ``stage1.anti_collapse.n_bloom_lookahead``
  must be exactly 0 in the default config, and ``threshold_upper`` must
  be null. Two independent 1000-epoch runs (symmetric plain-sum and
  asymmetric hinge-on-upper) regressed MAE by ~34%; the loss form is
  frozen at ADR-024 until a new ADR proves an alternative path.
* ADR-018 (Rejected V2.2) — ``stage1.curriculum.enabled`` must be ``false``
  in the default config. Hard-mask curriculum regressed MAE +39 % on ADR-027;
  code path retained for future ablation only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_CFG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
)


@pytest.fixture(scope="module")
def default_cfg() -> dict:
    assert _CFG_PATH.exists(), f"default config missing at {_CFG_PATH}"
    data = yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "default.yaml must parse to a mapping"
    return data


# ───────────────────────── ADR-020: full Löwdin disabled ─────────────────────


def test_adr020_full_lowdin_disabled_by_default(default_cfg: dict) -> None:
    """``use_full_lowdin_after_epoch`` must be ≥ 1_000_000 (sentinel).

    Per ADR-020, the eigh-based full Löwdin path is not safe as a routine
    training switch. Any value smaller than 1e6 risks the NaN cascade
    observed at epoch 201 of slice-1 patch.
    """
    threshold = (
        default_cfg.get("stage1", {})
        .get("ortho", {})
        .get("use_full_lowdin_after_epoch")
    )
    assert threshold is not None, (
        "stage1.ortho.use_full_lowdin_after_epoch missing from default.yaml "
        "— field must remain present for forward compatibility"
    )
    assert int(threshold) >= 1_000_000, (
        f"ADR-020 violation: use_full_lowdin_after_epoch={threshold} would "
        "re-enable the eigh-based full Löwdin path, which produced NaN "
        "gradients at epoch 201 in the slice-1 patch run. The field must "
        "remain ≥ 1_000_000 until eigh is replaced by a Cholesky-based "
        "whitening with documented gradient stability."
    )


def test_adr020_soft_ortho_weight_strong(default_cfg: dict) -> None:
    """``weights.ortho`` must be ≥ 50.0 to substitute for the disabled
    eigh-based hard projection."""
    w_ortho = default_cfg.get("stage1", {}).get("weights", {}).get("ortho")
    assert w_ortho is not None, "stage1.weights.ortho missing"
    assert float(w_ortho) >= 50.0, (
        f"ADR-020 corollary: weights.ortho={w_ortho} is too small. With "
        "full Löwdin disabled, the soft L_ortho penalty is the only "
        "cross-orbital orthogonality enforcer during stage 1, and must be "
        "weighted ≥ 50.0."
    )


# ───────────────────────── ADR-019: deprecated losses ────────────────────────


@pytest.mark.parametrize("name", ["node_pos", "node_cross", "sign"])
def test_adr019_deprecated_losses_have_zero_weight(
    default_cfg: dict, name: str
) -> None:
    """The three local shape-supervision losses must remain weight 0."""
    weights = default_cfg.get("stage1", {}).get("weights", {})
    assert name in weights, f"stage1.weights.{name} missing"
    assert float(weights[name]) == 0.0, (
        f"ADR-019 violation: weights.{name}={weights[name]} re-enables a "
        f"deprecated local shape-supervision loss."
    )


# ───────────────────────── ADR-021: anti-collapse loss ───────────────────────


def test_adr021_anti_collapse_weight_positive(default_cfg: dict) -> None:
    """``weights.anti_collapse`` must be strictly positive — disabling it
    re-opens the n-collapse attractor identified by the diagnostic."""
    weights = default_cfg.get("stage1", {}).get("weights", {})
    assert "anti_collapse" in weights, (
        "stage1.weights.anti_collapse missing — ADR-021 mandates this "
        "loss is active (re-add the key even if you intend to set it 0)."
    )
    assert float(weights["anti_collapse"]) > 0.0, (
        f"ADR-021 violation: weights.anti_collapse={weights['anti_collapse']} "
        "≤ 0. Re-running the n-collapse diagnostic and updating the ADR "
        "is required before disabling this loss."
    )


# ───────────────────────── ADR-023: anti_collapse hinge τ ────────────────────


def test_adr023_anti_collapse_threshold_in_bounds(default_cfg: dict) -> None:
    """``stage1.anti_collapse.threshold`` must be ∈ [0, 0.1]. The lower
    bound 0 recovers the ADR-021 plain-sum form (production default
    after ADR-024 revert). Above 0.1 the hinge tolerance is so loose
    that per-channel contamination of ≥10% goes unpenalised — defeats
    the entire point of ADR-023's mechanism."""
    block = default_cfg.get("stage1", {}).get("anti_collapse")
    assert block is not None, (
        "stage1.anti_collapse block missing — ADR-023 requires this "
        "section to exist (kept present even when τ=0 to make audit "
        "trail explicit)."
    )
    assert "threshold" in block, "stage1.anti_collapse.threshold missing"
    tau = float(block["threshold"])
    assert 0.0 <= tau <= 0.1 + 1.0e-9, (
        f"ADR-023 violation: anti_collapse.threshold={tau} not in [0, 0.1]. "
        "τ > 0.1 means each lower-n′ channel can carry ≥10% contamination "
        "without penalty, re-opening the failure mode that ADR-023 closed."
    )


# ───────────────────────── ADR-024: hinge revert (τ=0 default) ──────────────


def test_adr024_anti_collapse_threshold_default_is_zero(default_cfg: dict) -> None:
    """Production default for ``stage1.anti_collapse.threshold`` must be
    exactly 0 (plain-sum form). ADR-024 reverted the τ=0.02 hinge after
    a 1000-epoch run regressed MAE by +14% (1054 → 1201 meV). Setting
    τ > 0 in the default config requires an updated ADR-024 with the
    4-item re-enable checklist completed AND new training data showing
    the regression is closed (in particular: avg_smooth, avg_pde, AND
    per-level MAE breakdown, not just global avg_anti_collapse)."""
    tau = float(default_cfg["stage1"]["anti_collapse"]["threshold"])
    assert tau == 0.0, (
        f"ADR-024 violation: anti_collapse.threshold={tau} ≠ 0. The hinge "
        "form (τ > 0) was empirically regressed in the V2.1 1000-epoch "
        "run. Re-enabling requires the ADR-024 prerequisite checklist "
        "(bump smooth/c_norm, anneal τ, per-level MAE diagnostic, match "
        "weight against plain-sum baseline) and an ADR update with new "
        "evidence — not just a YAML edit."
    )


def test_adr024_anti_collapse_weight_at_or_above_five(default_cfg: dict) -> None:
    """ADR-024 keeps the F3 weight bump (3 → 5) while reverting the F4
    hinge. ``weights.anti_collapse`` must stay ≥ 5 so the plain-sum loss
    retains enough authority to push lower-n contamination below the
    levels reached by the original weight=3 plain-sum run (avg ≈ 0.49)."""
    w = float(default_cfg["stage1"]["weights"]["anti_collapse"])
    assert w >= 5.0 - 1.0e-9, (
        f"ADR-024 violation: weights.anti_collapse={w} < 5. The F3 weight "
        "bump (3 → 5) was preserved across the F4 revert because the "
        "regression evidence implicated the hinge form, not the higher "
        "weight. Lowering weight back to 3 requires a fresh 1000-epoch "
        "comparison and an ADR update."
    )


# ───────────── ADR-026 + ADR-028 (both Rejected): upper-window off ──────────


def test_adr026_anti_collapse_n_bloom_lookahead_present(default_cfg: dict) -> None:
    """The ``n_bloom_lookahead`` field must remain present in the default
    config so the YAML stays self-documenting (the comment explains why
    it's 0 and what would be required to flip it on again). This guards
    against silently removing the field when it's "just 0 anyway"."""
    sec = default_cfg.get("stage1", {}).get("anti_collapse", {})
    assert "n_bloom_lookahead" in sec, (
        "ADR-026 violation: stage1.anti_collapse.n_bloom_lookahead "
        "missing from default config. Keep the field present (with value "
        "0) so the Rejected status and re-enable preconditions stay "
        "visible in the YAML comment."
    )


def test_adr026_adr028_lookahead_is_zero_in_default(default_cfg: dict) -> None:
    """Default ``n_bloom_lookahead`` must be exactly 0.

    Two independent 1000-epoch runs (ADR-026 symmetric plain-sum
    upper window; ADR-028 asymmetric hinge-on-upper window) have
    both regressed MAE by ~34% — see "ADR-028 — Rejection rationale"
    in ``docs/design_rationale.md`` for the triple-failure summary
    (ADR-023 / 026 / 028). The anti_collapse loss is considered
    frozen at the ADR-024 baseline form (lower window only,
    plain-sum) until a new ADR proves it can escape that pattern.

    Re-enabling lookahead > 0 in the YAML — in *any* combination
    with threshold_upper — without a new ADR is a guardrail
    violation. This test enforces that gate."""
    la = int(default_cfg["stage1"]["anti_collapse"]["n_bloom_lookahead"])
    assert la == 0, (
        f"ADR-026/028 violation: n_bloom_lookahead={la} != 0. Both "
        "Accepted attempts at lookahead > 0 (ADR-026 symmetric plain-sum, "
        "ADR-028 asymmetric hinge) regressed MAE by ~34% in 1000-epoch "
        "validation. Re-enable requires a new ADR that explains why the "
        "ADR-023/026/028 'avg_anti_collapse improves but MAE regresses' "
        "anti-pattern won't reproduce — not a YAML edit alone."
    )


def test_adr028_threshold_upper_is_null_in_default(default_cfg: dict) -> None:
    """Default ``threshold_upper`` must be null (or absent).

    The field is only meaningful when ``n_bloom_lookahead > 0``,
    which itself is forbidden by the test above. A non-null value
    in the default config strongly suggests an in-progress attempt
    to bring lookahead back online without the required new ADR;
    this invariant is therefore strict (null only) even though the
    code path is a no-op while lookahead = 0."""
    sec = default_cfg["stage1"]["anti_collapse"]
    tu = sec.get("threshold_upper", None)
    assert tu is None, (
        f"ADR-028 (Rejected) violation: threshold_upper={tu!r} is not null. "
        "Per the ADR-028 revert, this field must stay null until a new "
        "ADR re-justifies lookahead > 0. Code path is retained for "
        "future A/B but the default config must not configure it."
    )


def test_adr026_anti_collapse_n_bloom_lookahead_within_cap(
    default_cfg: dict,
) -> None:
    """The helper hard-caps ``n_bloom_lookahead ≤ 3`` (grid extent
    rationale: ψ_an(Z, n', l) for n' > 6 doesn't fit r_max=50). The
    default trivially stays inside the cap while Rejected (=0), but
    this invariant is kept active so that an accidental re-enable via
    YAML edit also can't accidentally blow past the cap."""
    la = int(default_cfg["stage1"]["anti_collapse"]["n_bloom_lookahead"])
    assert la <= 3, (
        f"ADR-026 violation: n_bloom_lookahead={la} > 3. The cap is set "
        "in `losses/anti_collapse_loss.py::_MAX_N_BLOOM_LOOKAHEAD`. "
        "Going above 3 requires first extending the radial grid."
    )


# ───────────────────────── ADR-027: shape weight bump (R5) ──────────────────


def test_adr027_shape_weight_at_or_above_five(default_cfg: dict) -> None:
    """ADR-027 bumps ``stage1.weights.shape`` 2.0 → 5.0 to make the
    attractive ``L_shape`` term out-pull the Rayleigh drift that caused
    both the R3 n-collapse and the ADR-017 n-bloom. Lowering this back
    below 5.0 requires a fresh 1000-epoch comparison and an ADR update.
    """
    w = float(default_cfg["stage1"]["weights"]["shape"])
    assert w >= 5.0 - 1.0e-9, (
        f"ADR-027 violation: weights.shape={w} < 5.0. The 1000-epoch "
        "ADR-026 regression diagnosis showed the R3 attractor (weight=2) "
        "left avg_shape ≈ 0.47 — too soft, letting P drift to wrong-n. "
        "ADR-027 locks the attractor at weight=5. Re-evaluating "
        "downward requires evidence that avg_shape ≤ 0.30 was reached "
        "with the lower weight, or a stronger functional form replacing "
        "the linear 1−cos² L_shape."
    )


def test_adr027_shape_weight_pretrain_lockstep(default_cfg: dict) -> None:
    """``warmup.shape_weight_after_pretrain`` is the post-pretrain
    coefficient on the *same* ``L_shape``. ADR-027 requires the two
    fields to be exactly equal — any drift indicates the YAML was
    edited inconsistently (the trainer uses the pretrain value for
    epoch < shape_pretrain_epochs and the warmup value afterwards;
    keeping them equal means no behaviour change at the boundary)."""
    w_main = float(default_cfg["stage1"]["weights"]["shape"])
    w_post = float(
        default_cfg["stage1"]["warmup"]["shape_weight_after_pretrain"]
    )
    assert abs(w_main - w_post) < 1.0e-9, (
        f"ADR-027 lock-step violation: weights.shape={w_main} ≠ "
        f"warmup.shape_weight_after_pretrain={w_post}. These are two "
        "yaml fields driving the same L_shape coefficient (pretrain "
        "vs post-pretrain). Bumping one without the other creates a "
        "discontinuity at epoch = shape_pretrain_epochs. Update both "
        "or write an ADR explaining the intentional ramp."
    )


# ───────────────────────── ADR-018: n-curriculum ────────────────────────────


def test_adr018_curriculum_block_present(default_cfg: dict) -> None:
    """``stage1.curriculum`` must exist as a fully-spelled-out block.

    Even when curriculum is disabled (Rejected default), the block must
    remain so the self-documenting comment + schema invariants stay
    accurate. Removing the block silently would let a YAML edit revert
    to "no curriculum at all" without alarm.
    """
    sec = default_cfg.get("stage1", {})
    assert "curriculum" in sec, (
        "ADR-018 violation: stage1.curriculum missing. Keep the block "
        "present (set enabled: false rather than deleting the block)."
    )
    cur = sec["curriculum"]
    assert isinstance(cur, dict), (
        "stage1.curriculum must be a mapping (enabled + schedule + boost)."
    )
    for key in (
        "enabled",
        "schedule",
        "lambda_prior_boost",
        "lambda_prior_boost_decay_frac",
    ):
        assert key in cur, f"stage1.curriculum.{key} missing"


def test_adr018_curriculum_disabled_in_default(default_cfg: dict) -> None:
    """Default ``stage1.curriculum.enabled`` must be ``false``.

    ADR-018 hard-mask curriculum was empirically Rejected after a
    1000-epoch run on ADR-027 regressed MAE 934 → 1298 meV (+39 %),
    flipped (Z=3, n=2) from +4185 bloom to −7329 collapse, and
    produced a ×18 cliff at the 1a→1b transition. Re-enable requires
    a new ADR addressing 1a starvation — not a YAML edit alone."""
    enabled = default_cfg["stage1"]["curriculum"]["enabled"]
    assert enabled is False, (
        f"ADR-018 (Rejected) violation: curriculum.enabled={enabled!r}. "
        "Production default must stay false after the ADR-018 revert. "
        "Use an ablation config or new ADR to re-enable."
    )


def test_adr018_curriculum_schedule_fractions_sum_to_one(
    default_cfg: dict,
) -> None:
    """ADR-018: schedule fractions must sum to exactly 1.0 (within
    floating-point tolerance) so the sub-stages tile ``[0, n_epochs)``
    without gaps or overlaps."""
    schedule = default_cfg["stage1"]["curriculum"]["schedule"]
    fractions = [float(item["fraction"]) for item in schedule]
    total = sum(fractions)
    assert abs(total - 1.0) < 1.0e-6, (
        f"ADR-018 violation: curriculum.schedule fractions sum to {total} "
        f"(={fractions}), expected 1.0. _curriculum_state will mis-tile "
        "the stage1 epoch budget."
    )


def test_adr018_curriculum_n_max_active_monotone(default_cfg: dict) -> None:
    """ADR-018: each successive sub-stage must open *at least* as many
    n shells as the previous one (no closing of already-active shells)."""
    schedule = default_cfg["stage1"]["curriculum"]["schedule"]
    nmax = [int(item["n_max_active"]) for item in schedule]
    for i in range(1, len(nmax)):
        assert nmax[i] >= nmax[i - 1], (
            f"ADR-018 violation: sub-stage {i} closes active shells "
            f"(n_max_active {nmax[i - 1]} → {nmax[i]}). The curriculum "
            "must monotonically open more n, never close."
        )


def test_adr018_curriculum_lambda_prior_boost_bounds(default_cfg: dict) -> None:
    """ADR-018: ``lambda_prior_boost`` ∈ [1.0, 5.0]. < 1.0 makes the
    boost a decay (wrong direction); > 5.0 lets λ_prior dominate the
    loss at sub-stage entry and pins λ rigidly to Z/n."""
    boost = float(
        default_cfg["stage1"]["curriculum"]["lambda_prior_boost"]
    )
    assert 1.0 - 1e-9 <= boost <= 5.0 + 1e-9, (
        f"ADR-018 violation: lambda_prior_boost={boost} outside [1.0, 5.0]. "
        "Production default is 2.0."
    )


def test_adr018_curriculum_lambda_prior_boost_decay_frac_bounds(
    default_cfg: dict,
) -> None:
    """ADR-018: ``lambda_prior_boost_decay_frac`` ∈ (0, 1]. 0 collapses
    the decay window (no boost effect); > 1 lets the boost persist
    across the entire sub-stage, defeating the design."""
    decay = float(
        default_cfg["stage1"]["curriculum"]["lambda_prior_boost_decay_frac"]
    )
    assert 0.0 < decay <= 1.0 + 1e-9, (
        f"ADR-018 violation: lambda_prior_boost_decay_frac={decay} outside "
        "(0, 1]. Production default is 1/3."
    )


# ───────────────────────── ADR-022: tight lambda clamp ───────────────────────


def test_adr022_lam_log_res_clamp_tight(default_cfg: dict) -> None:
    """``envelope.lam_log_res_clamp`` must be ≤ 0.05 to remove the
    "envelope disguise window" for n-collapse."""
    clamp = default_cfg.get("envelope", {}).get("lam_log_res_clamp")
    assert clamp is not None, "envelope.lam_log_res_clamp missing"
    assert float(clamp) <= 0.05 + 1.0e-9, (
        f"ADR-022 violation: lam_log_res_clamp={clamp} > 0.05. The loose "
        "clamp allowed λ_pred to drift up to ±22% off Z/n, overlapping with "
        "λ_an(n-1) and enabling envelope-side n-collapse disguise."
    )


# ───────────────────────── ADR-017: per-sample normalisation ─────────────────


def test_adr017_envelope_lambda_min_present_and_safe(default_cfg: dict) -> None:
    """``envelope.lambda_min`` is the floor of the ADR-017 ``E_char²``.

    A missing or too-loose ``lambda_min`` lets the normalisation divisor
    underflow to zero in early training (before the lambda prior engages),
    which would either NaN the loss or spike its gradient. The trainer's
    ``_e_char_sq`` helper applies a ``max(λ_min, 1e-3)`` floor on top, so
    the explicit invariant is: ``lambda_min`` must exist and be ≥ 1e-3.
    """
    env = default_cfg.get("envelope", {})
    assert "lambda_min" in env, (
        "ADR-017 prerequisite: envelope.lambda_min missing. The per-sample "
        "normalisation in ADR-017 needs a defined lower bound on λ to floor "
        "E_char = λ²/2."
    )
    lam_min = float(env["lambda_min"])
    assert lam_min >= 1.0e-3, (
        f"ADR-017 prerequisite violation: envelope.lambda_min={lam_min} < 1e-3. "
        "E_char = λ²/2 would dip below 5e-7 Ha — the per-loss division "
        "clamp_min(1e-12) keeps things finite but the resulting normalised "
        "loss becomes pathologically large and dominates the gradient."
    )
