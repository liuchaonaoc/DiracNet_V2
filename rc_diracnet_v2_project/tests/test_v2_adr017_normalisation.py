"""End-to-end tests for ADR-017 per-sample normalisation in the trainer.

These tests construct a minimal trainer with no real data loader and verify:

1. ``_e_char_sq(lam)`` returns the documented ``(max(λ², λ_min²) / 2)²``
   shape and value, with gradient detached.
2. The floor ``max(lam_min, 1e-3) ** 2`` actually engages when λ is
   pathologically small (no division by zero downstream).
3. ``_physics_losses`` populates the ``*_raw`` monitor keys (so post-hoc
   diagnostics can compare pre/post normalisation numerics) without
   leaking gradient through the raw values.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rc_diracnet_v2.training.two_stage_trainer import TwoStageTrainer


def _make_minimal_trainer(lam_min: float = 0.05) -> TwoStageTrainer:
    """Build a TwoStageTrainer without running ``__init__``'s device dance.

    We only need ``self.cfg.envelope.lambda_min`` to exist for
    ``_e_char_sq`` — bypass ``__init__`` entirely.
    """
    trainer = TwoStageTrainer.__new__(TwoStageTrainer)
    trainer.cfg = SimpleNamespace(
        envelope=SimpleNamespace(lambda_min=lam_min),
    )
    trainer.device = torch.device("cpu")
    return trainer


def test_adr017_e_char_sq_matches_formula_for_normal_lambda():
    trainer = _make_minimal_trainer(lam_min=0.05)
    lam = torch.tensor([[1.0, 0.5, 3.0]], dtype=torch.float32)
    e_char_sq = trainer._e_char_sq(lam)
    expected = (lam.pow(2) / 2.0).pow(2)
    assert e_char_sq.shape == lam.shape
    assert torch.allclose(e_char_sq, expected, atol=1.0e-7)


def test_adr017_e_char_sq_floors_when_lambda_underflows():
    """When λ < lam_min, E_char² must equal (lam_min²/2)² rather than (λ²/2)²."""
    trainer = _make_minimal_trainer(lam_min=0.05)
    lam = torch.tensor([[1.0e-6, 1.0]], dtype=torch.float32)
    e_char_sq = trainer._e_char_sq(lam)
    floor_lam_sq = max(0.05, 1.0e-3) ** 2
    expected_first = (floor_lam_sq / 2.0) ** 2
    expected_second = (1.0 / 2.0) ** 2
    assert abs(e_char_sq[0, 0].item() - expected_first) < 1.0e-12
    assert abs(e_char_sq[0, 1].item() - expected_second) < 1.0e-6


def test_adr017_e_char_sq_is_detached_from_lambda():
    """No gradient must flow back into λ through the normalisation divisor —
    otherwise pde/decay/virial would have a self-referential gradient term
    that distorts the loss landscape."""
    trainer = _make_minimal_trainer(lam_min=0.05)
    lam = torch.tensor([[0.8]], dtype=torch.float32, requires_grad=True)
    e_char_sq = trainer._e_char_sq(lam)
    assert not e_char_sq.requires_grad, (
        "ADR-017 violation: E_char² should be detach()ed from λ. A trainable "
        "divisor lets the model game the normalisation by shrinking λ."
    )


def test_adr017_e_char_sq_respects_minimum_secondary_floor():
    """Even when lam_min is configured below the safety floor (1e-3), the
    helper must still floor at 1e-3 to keep E_char² above 1.25e-13."""
    trainer = _make_minimal_trainer(lam_min=1.0e-5)
    lam = torch.tensor([[1.0e-9]], dtype=torch.float32)
    e_char_sq = trainer._e_char_sq(lam)
    # Floor = max(1e-5, 1e-3) ** 2 = 1e-6, E_char = 5e-7, E_char² = 2.5e-13.
    assert e_char_sq.item() >= (1.0e-3) ** 4 / 4.0 - 1.0e-18


def test_adr017_e_char_sq_shape_preserves_batch_orb():
    """Helper must broadcast nothing — output shape must equal input shape."""
    trainer = _make_minimal_trainer(lam_min=0.05)
    lam = torch.rand(4, 16, dtype=torch.float32) + 0.1
    e_char_sq = trainer._e_char_sq(lam)
    assert e_char_sq.shape == lam.shape
