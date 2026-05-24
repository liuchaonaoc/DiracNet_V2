"""DiracNet V2 main model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..basis import BSplineBasis
from ..data.batch_builder import V2BatchBuilder
from ..data.level_encoder import LevelFeatureEncoder
from ..encoders.quantum_encoder import GlobalQuantumEncoder
from ..physics.dirac_operator import DiracRadialOperator, orbital_energy_from_dirac
from ..physics.slater_lda import compute_density
from ..readout import AnalyticEnvelope, BSplineReadout, lowdin_orthonormalize
from ..utils.grid import RadialGrid
from ..utils.numeric import sanitize_energies
from ..kan.mlp_coeff_net import MLPCoeffNet
from ..kan.kan_coeff_net import KANCoeffNet
from ..kan.pykan_coeff_net import PyKANCoeffNet
from .level_residual_head import LevelResidualHead


class DiracNetV2(nn.Module):
    def __init__(self, cfg, term_vocab_size: int = 256) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = RadialGrid(cfg.grid.r_min, cfg.grid.r_max, cfg.grid.n_grid, cfg.grid.scheme)
        self.encoder = GlobalQuantumEncoder(
            d_embed_z=cfg.encoder.d_embed_z,
            d_embed_shell=cfg.encoder.d_embed_shell,
            d_gru_hidden=cfg.encoder.d_gru_hidden,
            d_cond=cfg.encoder.d_cond,
            max_z=cfg.encoder.max_z,
            max_n=cfg.encoder.max_n,
            max_l=cfg.encoder.max_l,
            max_seq=cfg.encoder.max_seq,
        )
        self.level_encoder = LevelFeatureEncoder(cfg.encoder.d_cond, int(getattr(cfg.encoder, "term_vocab_size", term_vocab_size)))
        self.batch_builder = V2BatchBuilder(n_orb_max=cfg.readout.n_orb_max)
        self.bspline = BSplineBasis(cfg.bspline.n_basis, cfg.bspline.order, cfg.grid.r_min, cfg.grid.r_max, cfg.bspline.knot_scheme)
        self.bspline.precompute(self.grid.r)
        coeff_cls = _resolve_coeff_net(str(cfg.kan.kind))
        self.coeff_net = coeff_cls(
            d_cond=cfg.encoder.d_cond,
            d_phys=8,
            n_basis=cfg.bspline.n_basis,
            hidden=int(getattr(cfg.kan, "hidden", 128)),
            grid_size=int(getattr(cfg.kan, "grid_size", 9)),
            spline_order=int(getattr(cfg.kan, "spline_order", 3)),
            seed=int(getattr(cfg, "seed", 42)),
            max_z=cfg.encoder.max_z,
            max_n=cfg.encoder.max_n,
        )
        self.envelope = AnalyticEnvelope(use_relativistic_gamma=bool(getattr(cfg.envelope, "use_relativistic_gamma", False)))
        self.readout = BSplineReadout(force_c0_zero=bool(cfg.bspline.force_c0_zero), q_residual_scale=float(getattr(cfg.readout, "q_residual_scale", 0.0)))
        self.dirac_op = DiracRadialOperator()
        self.residual_head: LevelResidualHead | None = None
        if bool(getattr(cfg.stage2, "enabled", False)):
            self.residual_head = LevelResidualHead(cfg.encoder.d_cond, float(cfg.stage2.delta_max_meV))
        # Trainer-controlled flag; default False (norm-only Löwdin + soft ortho
        # loss). Late stage1 the trainer flips this to True so the symmetric
        # Löwdin path runs to wipe out residual cross-orbital overlap.
        self.use_full_lowdin_now: bool = False

    def forward(self, batch: dict[str, Any], use_residual: bool | None = None) -> dict[str, Any]:
        device = self.grid.r.device
        batch_d = _batch_to_device(batch, device)
        if "per_orb_features" not in batch_d:
            batch_d = self.batch_builder.build(batch_d)
        h = self.encoder(batch_d)
        h = self.level_encoder(h, batch_d["J"], batch_d["parity"], batch_d.get("term_id", torch.zeros_like(batch_d["J"])))
        orb_mask = batch_d["orb_mask"][:, : self.cfg.readout.n_orb_max]
        per_orb = batch_d["per_orb_features"][:, : self.cfg.readout.n_orb_max]
        lam_log_res, c = self.coeff_net(h, per_orb, orb_mask)
        if bool(getattr(self.cfg.envelope, "freeze_lambda_residual", False)):
            lam_log_res = torch.zeros_like(lam_log_res)
        z_eff = per_orb[..., 1].to(h.dtype).clamp_min(0.1)
        n_idx = per_orb[..., 2].to(h.dtype).clamp_min(1.0)
        lam_base = (z_eff / n_idx).clamp(float(self.cfg.envelope.lambda_min), float(self.cfg.envelope.lambda_max))
        clamp = float(getattr(self.cfg.envelope, "lam_log_res_clamp", 2.0))
        leak = float(getattr(self.cfg.envelope, "lam_log_res_clamp_leak", 0.05))
        # Leaky clamp: hard clamp zeros gradient on the boundary and locks every
        # excited-state lambda to lam_ref*exp(-clamp) (lambda_rel_err = 0.181269
        # when clamp=0.2). A small leak keeps lambda_prior able to pull lambda back.
        inside = lam_log_res.clamp(-clamp, clamp)
        lam_log_res_bounded = inside + leak * (lam_log_res - inside)
        lam = (lam_base * torch.exp(lam_log_res_bounded)).clamp(
            float(self.cfg.envelope.lambda_min), float(self.cfg.envelope.lambda_max)
        )
        kappa = batch_d["kappa"][:, : self.cfg.readout.n_orb_max]
        env, denv, d2env = self.envelope(self.grid.r, lam, kappa, batch_d["Z"])
        B, dB, d2B = self.bspline()
        wf = self.readout(c, env, denv, d2env, B, dB, d2B, kappa, self.grid.r, orb_mask)
        ortho = lowdin_orthonormalize(
            wf["P"], wf["Q"], self.grid, orb_mask, wf["dPdr"], wf["dQdr"],
            use_full_lowdin=bool(self.use_full_lowdin_now),
        )
        P, Q = ortho["P"], ortho["Q"]
        dPdr, dQdr = ortho["dPdr"], ortho["dQdr"]
        Z = batch_d["Z"].to(P.dtype)
        V_eff = -Z.view(-1, 1) / self.grid.r.view(1, -1).to(P.dtype)
        LP, LQ = self.dirac_op.apply(P, Q, dPdr, dQdr, V_eff, kappa, self.grid.r)
        E_orb = sanitize_energies(orbital_energy_from_dirac(P, Q, LP, LQ, self.grid), clamp_finite=False)
        E_orb = torch.where(orb_mask, E_orb, torch.zeros_like(E_orb))
        occ = per_orb[..., 6].to(E_orb.dtype)
        E_orb_sum = (E_orb * occ * orb_mask.to(E_orb.dtype)).sum(dim=-1)
        want_res = bool(use_residual) if use_residual is not None else bool(getattr(self.cfg.stage2, "enabled", False))
        delta = None
        if want_res and self.residual_head is not None:
            delta = self.residual_head(h)
        E_cal = E_orb_sum if delta is None else E_orb_sum + delta
        rho = compute_density(P, Q, occ, self.grid)
        return {
            "h_cond": h,
            "per_orb_features": per_orb,
            "lam": lam,
            "lam_log_res": lam_log_res,
            "c_raw": c,
            "wavefunctions": {"P": P, "Q": Q, "dPdr": dPdr, "dQdr": dQdr},
            "factor": {"f": wf["f"], "df": wf["df"], "d2f": wf["d2f"]},
            "density": rho,
            "v_eff": V_eff,
            "E_orb": E_orb,
            "macro": {"E_orb_sum": E_orb_sum, "E_pred_main": E_orb_sum, "delta_residual": delta, "E_pred_calibrated": E_cal},
        }


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _resolve_coeff_net(kind: str):
    if kind == "kan":
        return KANCoeffNet
    if kind == "pykan":
        return PyKANCoeffNet
    if kind == "mlp":
        return MLPCoeffNet
    raise ValueError(f"unknown kan.kind={kind!r}; expected 'kan', 'pykan', or 'mlp'")
