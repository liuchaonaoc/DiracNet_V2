"""Sanity tests for the warmup+cosine scheduler with min_lr_ratio floor.

The previous regression (MAE 1831 meV → 8678 meV) was driven by cosine decay
hitting lr=0 well before training converged. The floor prevents that and these
tests pin the behaviour so we don't regress again.
"""

from __future__ import annotations

import torch

from rc_diracnet_v2.training.scheduler import build_scheduler


class _Cfg:
    def __init__(self, **kw):
        self.scheduler = type("Sched", (), kw)()


def _lr_trajectory(total, warmup, min_ratio, peak=1.0e-4):
    cfg = _Cfg(
        name="warmup_cosine",
        warmup_steps=warmup,
        total_steps=total,
        min_lr_ratio=min_ratio,
    )
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=peak)
    sched = build_scheduler(opt, cfg)
    lrs = []
    for _ in range(total + 5):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_min_lr_ratio_floor_holds_after_full_decay():
    """At step >= total_steps, lr must equal min_lr_ratio * peak (not 0)."""
    lrs = _lr_trajectory(total=100, warmup=10, min_ratio=0.1, peak=1.0e-4)
    assert lrs[-1] == lrs[-2] == lrs[-3]
    assert abs(lrs[-1] - 1.0e-5) < 1.0e-12


def test_min_lr_ratio_zero_decays_to_zero():
    """Backward compat: min_lr_ratio=0 (or missing) reaches lr=0 at the end."""
    lrs = _lr_trajectory(total=100, warmup=10, min_ratio=0.0, peak=1.0e-4)
    assert lrs[-1] < 1.0e-12


def test_warmup_ramps_from_zero_to_peak():
    lrs = _lr_trajectory(total=100, warmup=10, min_ratio=0.1, peak=1.0e-4)
    assert lrs[0] == 0.0
    assert abs(lrs[10] - 1.0e-4) < 1.0e-12  # end of warmup hits peak
    assert lrs[5] < lrs[9]  # monotone during warmup


def test_post_warmup_lr_is_monotone_non_increasing():
    lrs = _lr_trajectory(total=200, warmup=20, min_ratio=0.1, peak=1.0e-4)
    for i in range(20, len(lrs) - 1):
        assert lrs[i] + 1.0e-12 >= lrs[i + 1]
