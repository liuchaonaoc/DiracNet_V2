from pathlib import Path

import torch

from rc_diracnet_v2.models import DiracNetV2
from scripts._common import load_cfg


def test_full_forward_pass(hydrogenic_h_1s_batch):
    cfg = load_cfg(Path("configs/default.yaml"))
    model = DiracNetV2(cfg)
    out = model(hydrogenic_h_1s_batch)
    assert out["E_orb"].shape[0] == 1


def _leaky_clamp(x: torch.Tensor, clamp: float, leak: float) -> torch.Tensor:
    inside = x.clamp(-clamp, clamp)
    return inside + leak * (x - inside)


def test_leaky_clamp_is_identity_inside_window(hydrogenic_h_1s_batch):
    cfg = load_cfg(Path("configs/default.yaml"))
    clamp = float(cfg.envelope.lam_log_res_clamp)
    assert clamp <= 1.0, "expected lam_log_res_clamp tightened to <= 1.0"
    x = torch.linspace(-clamp + 1e-3, clamp - 1e-3, 11)
    y = _leaky_clamp(x, clamp, leak=0.05)
    assert torch.allclose(x, y, atol=1e-7)


def test_leaky_clamp_keeps_gradient_alive_at_extreme_residuals():
    """Hard clamp locks lambda at the boundary; leak must preserve gradient."""
    cfg = load_cfg(Path("configs/default.yaml"))
    clamp = float(cfg.envelope.lam_log_res_clamp)
    leak = float(getattr(cfg.envelope, "lam_log_res_clamp_leak", 0.05))
    for extreme in [-50.0, -5.0, 5.0, 50.0]:
        x = torch.tensor([extreme], requires_grad=True)
        y = _leaky_clamp(x, clamp, leak)
        (g,) = torch.autograd.grad(y.sum(), x)
        assert abs(float(g) - leak) < 1e-6, f"expected leak={leak} at x={extreme}, got {float(g)}"


def test_real_forward_keeps_lam_close_to_z_over_n(hydrogenic_h_1s_batch):
    cfg = load_cfg(Path("configs/default.yaml"))
    clamp = float(cfg.envelope.lam_log_res_clamp)
    model = DiracNetV2(cfg)
    out = model(hydrogenic_h_1s_batch)
    lam = out["lam"]
    orb_mask = hydrogenic_h_1s_batch["orb_mask"][:, : cfg.readout.n_orb_max]
    # Reference lam = Z / n for the active 1s orbital (Z=1, n=1).
    lam_active = lam[orb_mask]
    ratio = lam_active / 1.0
    bound = float(torch.exp(torch.tensor(clamp))) + 1e-4
    assert torch.all(ratio <= bound) and torch.all(ratio >= 1.0 / bound)
