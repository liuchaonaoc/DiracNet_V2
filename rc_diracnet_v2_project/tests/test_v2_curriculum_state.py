"""ADR-018 — Unit tests for the n-curriculum state helper.

``TwoStageTrainer._curriculum_state(epoch, n_epochs, cur_cfg)`` is a pure
function that returns ``(active_n_max, lambda_prior_boost, sub_stage_name)``
for a given stage-1 epoch. These tests pin its behaviour at the
operationally critical points:

* Disabled / empty-schedule → no-op sentinel
* Sub-stage boundary epochs map to the correct sub-stage
* λ_prior boost starts at ``boost_max`` on sub-stage entry and linearly
  decays to 1.0 within ``decay_frac × sub_len`` epochs
* `n_max_active` is monotone non-decreasing across the schedule
* Smoke (n_epochs=20) and 1000-epoch validation use the same fractions

The function is purposely a staticmethod with all dependencies passed
explicitly, so these tests don't need a Trainer instance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rc_diracnet_v2.training.two_stage_trainer import TwoStageTrainer


_state = TwoStageTrainer._curriculum_state


def _make_cfg(
    enabled: bool = True,
    schedule: list[dict] | None = None,
    boost: float = 2.0,
    decay_frac: float = 1.0 / 3.0,
) -> SimpleNamespace:
    """Build a SimpleNamespace cfg block as ``_curriculum_state`` expects.

    The helper accepts both dict-items and attribute-style items in
    ``schedule``; tests here use dict-items, matching the YAML shape.
    """
    if schedule is None:
        schedule = [
            {"name": "1a", "fraction": 0.20, "n_max_active": 1},
            {"name": "1b", "fraction": 0.25, "n_max_active": 2},
            {"name": "1c", "fraction": 0.30, "n_max_active": 3},
            {"name": "1d", "fraction": 0.25, "n_max_active": 99},
        ]
    return SimpleNamespace(
        enabled=enabled,
        schedule=schedule,
        lambda_prior_boost=boost,
        lambda_prior_boost_decay_frac=decay_frac,
    )


# ───────────────────────── disabled / no-op paths ───────────────────────────


def test_curriculum_disabled_returns_open_all_sentinel():
    cfg = _make_cfg(enabled=False)
    n_max, boost, name = _state(epoch=0, n_epochs=1000, cur_cfg=cfg)
    assert n_max >= 10 ** 5, "disabled curriculum must return open-all sentinel"
    assert boost == 1.0
    assert name == "off"


def test_curriculum_none_cfg_is_noop():
    n_max, boost, name = _state(epoch=42, n_epochs=1000, cur_cfg=None)
    assert n_max >= 10 ** 5
    assert boost == 1.0
    assert name == "off"


def test_curriculum_empty_schedule_is_noop():
    cfg = _make_cfg(schedule=[])
    n_max, boost, name = _state(epoch=0, n_epochs=100, cur_cfg=cfg)
    assert n_max >= 10 ** 5
    assert boost == 1.0
    assert name == "off"


# ───────────────────── sub-stage boundary mapping (1000-epoch) ──────────────


@pytest.mark.parametrize(
    "epoch,expected_name,expected_n_max",
    [
        (0, "1a", 1),
        (1, "1a", 1),
        (199, "1a", 1),
        (200, "1b", 2),
        (449, "1b", 2),
        (450, "1c", 3),
        (749, "1c", 3),
        (750, "1d", 99),
        (999, "1d", 99),
    ],
)
def test_curriculum_boundaries_1000_epoch(epoch, expected_name, expected_n_max):
    """ADR-018 §Decision: 200+250+300+250 in the 1000-epoch case."""
    cfg = _make_cfg()
    n_max, _boost, name = _state(epoch=epoch, n_epochs=1000, cur_cfg=cfg)
    assert name == expected_name
    assert n_max == expected_n_max


# ─────────────────────── sub-stage mapping (20-epoch smoke) ─────────────────


@pytest.mark.parametrize(
    "epoch,expected_name",
    [
        (0, "1a"),       # 0.20 * 20 = 4 → 1a covers [0, 4)
        (3, "1a"),
        (4, "1b"),       # 0.25 * 20 = 5 → 1b covers [4, 9)
        (8, "1b"),
        (9, "1c"),       # 0.30 * 20 = 6 → 1c covers [9, 15)
        (14, "1c"),
        (15, "1d"),      # 1d covers [15, 20)
        (19, "1d"),
    ],
)
def test_curriculum_boundaries_20_epoch_smoke(epoch, expected_name):
    """Same fractions must give a sensible 20-epoch breakdown
    (4 + 5 + 6 + 5 = 20)."""
    cfg = _make_cfg()
    _n_max, _boost, name = _state(epoch=epoch, n_epochs=20, cur_cfg=cfg)
    assert name == expected_name


# ─────────────────────────── λ_prior boost behaviour ────────────────────────


def test_curriculum_boost_max_on_entry():
    """At sub-stage entry epoch, boost == boost_max (= 2.0 by default)."""
    cfg = _make_cfg()
    for entry_epoch in (0, 200, 450, 750):
        _, boost, _ = _state(epoch=entry_epoch, n_epochs=1000, cur_cfg=cfg)
        assert boost == pytest.approx(2.0, abs=1e-9), (
            f"entry epoch {entry_epoch}: expected boost=2.0, got {boost}"
        )


def test_curriculum_boost_decays_to_one():
    """After ``decay_frac × sub_len`` epochs, boost == 1.0."""
    cfg = _make_cfg()
    # Sub-stage 1a covers [0, 200), decay window = round(0.333 * 200) = 67.
    # By epoch 67 onwards within 1a, boost must be exactly 1.0.
    for ep in (67, 100, 150, 199):
        _, boost, name = _state(epoch=ep, n_epochs=1000, cur_cfg=cfg)
        assert name == "1a"
        assert boost == pytest.approx(1.0, abs=1e-9), (
            f"epoch {ep}: post-decay boost must be 1.0, got {boost}"
        )


def test_curriculum_boost_is_monotone_decreasing_within_window():
    """Within the decay window, boost monotonically decreases from
    boost_max toward 1.0."""
    cfg = _make_cfg()
    # 1b entry at 200, decay window = round(0.333 * 250) = 83
    boosts = [
        _state(epoch=ep, n_epochs=1000, cur_cfg=cfg)[1]
        for ep in range(200, 200 + 83)
    ]
    # Must start at 2.0 and end approaching 1.0
    assert boosts[0] == pytest.approx(2.0, abs=1e-9)
    assert boosts[-1] >= 1.0 - 1e-9
    # Monotone non-increasing
    for prev, nxt in zip(boosts, boosts[1:]):
        assert nxt <= prev + 1e-9, (
            f"boost should be non-increasing within decay window: "
            f"{prev} → {nxt}"
        )


def test_curriculum_boost_is_silent_when_boost_eq_one():
    """``lambda_prior_boost = 1.0`` ⇒ boost identically 1.0 everywhere."""
    cfg = _make_cfg(boost=1.0)
    for ep in (0, 50, 100, 199, 200, 449, 450, 750, 999):
        _, boost, _ = _state(epoch=ep, n_epochs=1000, cur_cfg=cfg)
        assert boost == pytest.approx(1.0, abs=1e-9), (
            f"boost=1.0 cfg should keep boost flat at 1.0; got {boost} @ {ep}"
        )


# ───────────────────────── monotonicity invariants ──────────────────────────


def test_curriculum_active_n_max_is_monotone_non_decreasing():
    """Stepping through every epoch must yield a non-decreasing
    ``active_n_max`` sequence."""
    cfg = _make_cfg()
    prev_n_max = -1
    for ep in range(1000):
        n_max, _, _ = _state(epoch=ep, n_epochs=1000, cur_cfg=cfg)
        assert n_max >= prev_n_max, (
            f"epoch {ep}: active_n_max dropped {prev_n_max} → {n_max} "
            "(violates ADR-018 'open more n' semantics)"
        )
        prev_n_max = n_max


def test_curriculum_last_sub_stage_absorbs_rounding_slack():
    """If fractions don't divide n_epochs cleanly, the last sub-stage
    must still cover up to (but not including) ``n_epochs``."""
    cfg = _make_cfg()
    # n_epochs = 17 → rough breakdown 3 + 4 + 5 + ? (5 to reach 17)
    _, _, name_last = _state(epoch=16, n_epochs=17, cur_cfg=cfg)
    assert name_last == "1d", "last epoch must land in the last sub-stage"


def test_curriculum_short_run_still_assigns_every_epoch():
    """n_epochs = 4 (4 sub-stages → 1 epoch each) — every epoch must
    map to some sub-stage, no IndexError."""
    cfg = _make_cfg()
    seen = set()
    for ep in range(4):
        _, _, name = _state(epoch=ep, n_epochs=4, cur_cfg=cfg)
        seen.add(name)
    assert len(seen) >= 1


def test_curriculum_handles_attribute_style_schedule():
    """``_curriculum_state`` should accept SimpleNamespace items in
    addition to dict items (matches OmegaConf's behaviour)."""
    schedule = [
        SimpleNamespace(name="1a", fraction=0.5, n_max_active=2),
        SimpleNamespace(name="1b", fraction=0.5, n_max_active=99),
    ]
    cfg = _make_cfg(schedule=schedule)
    n_max_first, _, name_first = _state(epoch=0, n_epochs=10, cur_cfg=cfg)
    n_max_last, _, name_last = _state(epoch=9, n_epochs=10, cur_cfg=cfg)
    assert name_first == "1a" and n_max_first == 2
    assert name_last == "1b" and n_max_last == 99
