"""Unit tests for the action-loss warmup schedule (§15 slice 1).

The Bohr-Sommerfeld action loss is opened from weight=0 by an epoch-based
linear ramp, replacing the previous ``L_PDE < threshold`` gate which never
fires after ``DiracPDELoss(detach_energy=True)`` (PDE residual stabilises
around 1.0–1.5 in our setting).
"""

from __future__ import annotations

import math

import pytest

from rc_diracnet_v2.training.two_stage_trainer import TwoStageTrainer


_ramp = TwoStageTrainer._action_warmup_weight


def test_action_warmup_disabled_when_weight_zero():
    assert _ramp(epoch=0, weight_full=0.0, start_epoch=30, end_epoch=100) == 0.0
    assert _ramp(epoch=999, weight_full=0.0, start_epoch=30, end_epoch=100) == 0.0


def test_action_warmup_zero_before_start():
    for ep in (0, 1, 15, 29):
        assert _ramp(epoch=ep, weight_full=0.5, start_epoch=30, end_epoch=100) == 0.0


def test_action_warmup_full_at_and_after_end():
    for ep in (100, 101, 500, 1000):
        assert _ramp(epoch=ep, weight_full=0.5, start_epoch=30, end_epoch=100) == 0.5


def test_action_warmup_linear_inside_ramp():
    w_full = 0.5
    start, end = 30, 100
    cases = [
        (30, 0.0),
        (45, w_full * (45 - 30) / (end - start)),
        (65, w_full * (65 - 30) / (end - start)),
        (99, w_full * (99 - 30) / (end - start)),
    ]
    for ep, expected in cases:
        got = _ramp(epoch=ep, weight_full=w_full, start_epoch=start, end_epoch=end)
        assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9), (
            f"epoch={ep}: got {got}, expected {expected}"
        )


def test_action_warmup_monotonic_non_decreasing():
    w_full = 0.5
    start, end = 30, 100
    prev = -1.0
    for ep in range(0, 200):
        cur = _ramp(epoch=ep, weight_full=w_full, start_epoch=start, end_epoch=end)
        assert cur >= prev - 1e-12, f"non-monotonic at epoch {ep}: {prev} -> {cur}"
        prev = cur


def test_action_warmup_handles_degenerate_window():
    """If end == start, the schedule should jump straight to full weight at start."""
    assert _ramp(epoch=29, weight_full=0.5, start_epoch=30, end_epoch=30) == 0.0
    assert _ramp(epoch=30, weight_full=0.5, start_epoch=30, end_epoch=30) == 0.5
    assert _ramp(epoch=31, weight_full=0.5, start_epoch=30, end_epoch=30) == 0.5


def test_action_warmup_negative_weight_full_treated_as_disabled():
    """Defensive: a misconfigured negative weight should not flow into backward."""
    assert _ramp(epoch=50, weight_full=-1.0, start_epoch=30, end_epoch=100) == 0.0
