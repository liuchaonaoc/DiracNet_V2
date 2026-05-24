"""Analytic physics gate and residual safety verdicts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import HARTREE_eV
from ..physics.hydrogenic_analytic import cosine_signed, hydrogenic_energy, hydrogenic_P_analytic


@dataclass
class GateReport:
    passed: bool
    cos_min: float
    e_err_max_mev: float
    lambda_rel_max: float
    L_action_BS: float | None = None

    def to_markdown(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"""# Stage Gate Report\n\n- verdict: {verdict}\n- cos_min: {self.cos_min:.6f}\n- e_err_max_mev: {self.e_err_max_mev:.3f}\n- lambda_rel_max: {self.lambda_rel_max:.6f}\n- L_action_BS: {self.L_action_BS}\n"""


def check_stage1_gate(model, loader, thresholds) -> GateReport:
    model.eval()
    cos_vals, e_errs, lam_errs = [], [], []
    with torch.no_grad():
        for batch in loader:
            device = next(model.parameters()).device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch, use_residual=False)
            P = out["wavefunctions"]["P"][:, 0]
            E = out["E_orb"][:, 0]
            lam = out["lam"][:, 0]
            cfg = batch["config_shells"][:, 0]
            for i in range(P.shape[0]):
                Z = int(batch["Z"][i].item())
                n = int(cfg[i, 0].item())
                l = int(cfg[i, 1].item())
                pref = hydrogenic_P_analytic(model.grid.r.to(P.device), Z, n, l)
                cos_vals.append(float(cosine_signed(P[i], pref, model.grid).abs().item()))
                eref = hydrogenic_energy(Z, n)
                e_errs.append(abs(float(E[i].item()) - eref) * HARTREE_eV * 1000.0)
                lam_ref = Z / max(n, 1)
                lam_errs.append(abs(float(lam[i].item()) - lam_ref) / max(lam_ref, 1e-12))
    cos_min = min(cos_vals) if cos_vals else 0.0
    emax = max(e_errs) if e_errs else float("inf")
    lmax = max(lam_errs) if lam_errs else float("inf")
    passed = (
        cos_min >= float(thresholds.cos_threshold)
        and emax <= float(thresholds.e_orb_meV_threshold)
        and lmax <= float(thresholds.lambda_rel_threshold)
    )
    return GateReport(passed, cos_min, emax, lmax)


def residual_safety_verdict(metrics: dict[str, float]) -> str:
    if not metrics.get("physics_gate_pass", False):
        return "CHEATING_RISK"
    if metrics.get("delta_ratio_median", 0.0) > metrics.get("delta_ratio_threshold", 0.2):
        return "CHEATING_RISK"
    cap = metrics.get("delta_cap_mev", float("inf"))
    if metrics.get("delta_abs_max_mev", 0.0) > 0.9 * cap:
        return "CHEATING_RISK"
    if metrics.get("loo_rms_mev", 0.0) > metrics.get("loo_rms_threshold_mev", 200.0):
        return "CHEATING_RISK"
    if metrics.get("e_orb_only_mae_mev", 0.0) > metrics.get("e_orb_only_required_mae_mev", 50.0):
        return "CALIBRATION_ONLY_NOT_PHYSICS_PASS"
    return "ACCEPT_CALIBRATION"
