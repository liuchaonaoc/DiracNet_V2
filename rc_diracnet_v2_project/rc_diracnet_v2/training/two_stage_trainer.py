"""Two-stage trainer skeleton for V2."""

from __future__ import annotations

from typing import Any

import torch

from ..losses import (
    AntiCollapseLoss,
    AsymptoticTailLoss,
    BohrSommerfeldActionLoss,
    BSplineSmoothLoss,
    DecayConsistencyLoss,
    DiracPDELoss,
    NISTScalarHuberLoss,
    NodeCountLoss,
    OrthonormalityLoss,
    VirialLoss,
)
from ..physics.hydrogenic_analytic import (
    hydrogenic_P_analytic,
    hydrogenic_lobe_partition,
    hydrogenic_radial_nodes,
)
from ._helpers import to_device
from .optimizer_factory import build_optimizer
from .scheduler import build_scheduler


class TwoStageTrainer:
    def __init__(self, model, train_loader, val_loader, cfg) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu") if hasattr(cfg, "get") else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def _e_char_sq(self, lam: torch.Tensor) -> torch.Tensor:
        """ADR-017 — characteristic energy² for per-sample normalisation.

        Returns ``E_char² = (max(λ², λ_min²) / 2)²`` as a ``[B, N_orb]``
        tensor with gradient detached. The floor on ``λ²`` prevents the
        early-training divergence when ``λ`` momentarily underflows the
        envelope lower bound (the prior may pull λ small before
        ``L_lambda_prior`` engages). ``E_char²`` itself does NOT receive a
        ``clamp_min`` — that is left to the per-loss division so the
        clamp value can stay loss-local and well-tested.

        Source: ADR-017 "Per-sample 相对化归一化" §Decision.
        """
        lam_min = float(getattr(self.cfg.envelope, "lambda_min", 0.05))
        lam_floor_sq = max(lam_min, 1.0e-3) ** 2
        e_char = lam.detach().pow(2).clamp_min(lam_floor_sq) / 2.0
        return e_char.pow(2)

    def _physics_losses(self, batch: dict[str, Any], out: dict[str, Any]) -> dict[str, torch.Tensor]:
        wf = out["wavefunctions"]
        orb_mask = batch["orb_mask"].to(self.device)[:, : self.cfg.readout.n_orb_max]
        per = out["per_orb_features"]
        n_idx = per[..., 2].long()
        l_idx = per[..., 4].long()

        # ADR-018 — n-curriculum mask. ``_cur_active_n_max`` is set on
        # ``self`` by ``run_stage1`` *before* every batch via
        # ``_curriculum_state``. Default sentinel (10**6) is a no-op,
        # so callers that don't set up curriculum (older tests, stage-2
        # NIST calibration) see identical behaviour as before ADR-018.
        cur_n_max = int(getattr(self, "_cur_active_n_max", 10 ** 6))
        if cur_n_max < 10 ** 5:
            # Zero out every orbital with n > active_n_max so it cannot
            # contribute to any downstream loss's backward. Applies uniformly
            # to pde / ortho / shape / lambda_prior / anti_collapse / etc.
            # — every loss in this function consumes ``orb_mask``.
            curriculum_keep = (n_idx <= cur_n_max).to(orb_mask.dtype)
            orb_mask = orb_mask * curriculum_keep
        z_eff = per[..., 1].to(out["lam"].dtype).clamp_min(1.0e-6)
        n_float = per[..., 2].to(out["lam"].dtype).clamp_min(1.0)
        lam_ref = (z_eff / n_float).clamp(float(self.cfg.envelope.lambda_min), float(self.cfg.envelope.lambda_max))
        kappa_b = batch["kappa"].to(self.device)[:, : self.cfg.readout.n_orb_max]

        # ADR-017: shared E_char² for the three energy-dimensioned losses.
        # The same tensor is used by pde / decay_consistency / virial so they
        # remain in proportion to each other across (Z, n).
        e_char_sq = self._e_char_sq(out["lam"])

        # Backward path — normalised losses (the values the optimiser sees).
        pde_loss = DiracPDELoss(detach_energy=True)(
            wf["P"], wf["Q"], wf["dPdr"], wf["dQdr"], out["E_orb"],
            out["v_eff"], kappa_b, self.model.grid.r, self.model.grid, orb_mask,
            e_char_sq=e_char_sq,
        )
        decay_loss = DecayConsistencyLoss()(
            out["lam"], out["E_orb"], orb_mask, e_char_sq=e_char_sq,
        )
        virial_loss = VirialLoss()(
            wf["P"], wf["Q"], out["E_orb"], out["v_eff"],
            self.model.grid.r, self.model.grid, orb_mask,
            e_char_sq=e_char_sq,
        )

        # ADR-017 diagnostic monitors — pre-normalisation values so we can
        # compare numerics against ADR-017-disabled runs. Computed under
        # no_grad to avoid double backward memory; values are floats only.
        with torch.no_grad():
            pde_raw = DiracPDELoss(detach_energy=True)(
                wf["P"], wf["Q"], wf["dPdr"], wf["dQdr"], out["E_orb"],
                out["v_eff"], kappa_b, self.model.grid.r, self.model.grid, orb_mask,
                e_char_sq=None,
            )
            decay_raw = DecayConsistencyLoss()(
                out["lam"], out["E_orb"], orb_mask, e_char_sq=None,
            )
            virial_raw = VirialLoss()(
                wf["P"], wf["Q"], out["E_orb"], out["v_eff"],
                self.model.grid.r, self.model.grid, orb_mask,
                e_char_sq=None,
            )

        # ADR-028 — anti_collapse params read once so the dict literal
        # below stays a pure expression.
        _ac_cfg = getattr(self.cfg.stage1, "anti_collapse", None)
        _ac_thr_upper = getattr(_ac_cfg, "threshold_upper", None)
        anti_collapse_loss_fn = AntiCollapseLoss(
            self.model.grid,
            threshold=float(getattr(_ac_cfg, "threshold", 0.02)),
            n_bloom_lookahead=int(getattr(_ac_cfg, "n_bloom_lookahead", 0)),
            threshold_upper=(
                None if _ac_thr_upper is None else float(_ac_thr_upper)
            ),
        )

        return {
            "pde": pde_loss,
            "pde_raw": pde_raw,
            "ortho": OrthonormalityLoss()(wf["P"], wf["Q"], self.model.grid, orb_mask),
            "node": NodeCountLoss()(wf["P"], (n_idx - l_idx - 1).clamp_min(0), orb_mask),
            "node_pos": self._hydrogenic_node_position_loss(batch, wf["P"], orb_mask),
            "node_cross": self._hydrogenic_node_crossing_loss(batch, wf["P"], orb_mask),
            "lobe_ratio": self._hydrogenic_lobe_ratio_loss(batch, wf["P"], orb_mask),
            "sign": self._hydrogenic_sign_loss(batch, wf["P"], orb_mask),
            "action": BohrSommerfeldActionLoss()(out["E_orb"], out["v_eff"], n_idx, l_idx, orb_mask, self.model.grid),
            "asym": AsymptoticTailLoss()(wf["P"], self.model.grid.r, orb_mask),
            "smooth": BSplineSmoothLoss()(out["c_raw"], orb_mask),
            "lambda_prior": self._lambda_prior_loss(out["lam"], lam_ref, orb_mask),
            "shape": self._hydrogenic_shape_loss(batch, wf["P"], orb_mask),
            "factor_amp": self._factor_amplitude_loss(out["factor"]["f"], orb_mask),
            "c_norm": self._c_norm_loss(out["c_raw"], orb_mask),
            "decay_consistency": decay_loss,
            "decay_consistency_raw": decay_raw,
            "virial": virial_loss,
            "virial_raw": virial_raw,
            # ADR-021 + ADR-024 + ADR-028: anti-(n-collapse + n-bloom) loss.
            # Constructor is built above as ``anti_collapse_loss_fn`` so the
            # dict literal stays a pure expression. See ADR-028 for the
            # asymmetric (lower plain-sum, upper hinge) rationale.
            "anti_collapse": anti_collapse_loss_fn(
                wf["P"], batch["config_shells"].to(wf["P"].device),
                batch["Z"].to(wf["P"].device), orb_mask,
                nele=batch.get("nele"),
            ),
        }

    def _lambda_prior_loss(self, lam: torch.Tensor, lam_ref: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        err = torch.log(lam.clamp_min(1.0e-8)) - torch.log(lam_ref.clamp_min(1.0e-8))
        mask = orb_mask.to(lam.device).to(err.dtype)
        return (err.pow(2) * mask).sum() / mask.sum().clamp_min(1.0)

    def _hydrogenic_shape_loss(self, batch: dict[str, Any], P: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        one_electron = batch.get("nele")
        if one_electron is None:
            return P.sum() * 0.0
        config = batch["config_shells"].to(P.device)
        Z = batch["Z"].to(P.device)
        nele_mask = one_electron.to(P.device).view(-1, 1) <= 1
        losses: list[torch.Tensor] = []
        max_orb = min(P.shape[1], config.shape[1])
        for b in range(P.shape[0]):
            for o in range(max_orb):
                if not (bool(orb_mask[b, o].item()) and bool(nele_mask[b, 0].item())):
                    continue
                n = int(config[b, o, 0].item())
                l = int(config[b, o, 1].item())
                if n <= l or n <= 0:
                    continue
                pref = hydrogenic_P_analytic(self.model.grid.r.to(P.device), int(Z[b].item()), n, l)
                p = P[b, o]
                num = self.model.grid.integrate(p * pref, dim=-1)
                den = (
                    self.model.grid.integrate(p * p, dim=-1).sqrt()
                    * self.model.grid.integrate(pref * pref, dim=-1).sqrt()
                ).clamp_min(1.0e-12)
                cos = num / den
                losses.append(1.0 - cos.pow(2))
        if not losses:
            return P.sum() * 0.0
        return torch.stack(losses).mean()

    def _hydrogenic_node_position_loss(self, batch: dict[str, Any], P: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        one_electron = batch.get("nele")
        if one_electron is None:
            return P.sum() * 0.0
        config = batch["config_shells"].to(P.device)
        Z = batch["Z"].to(P.device)
        nele_mask = one_electron.to(P.device).view(-1, 1) <= 1
        r = self.model.grid.r.to(P.device).to(P.dtype)
        losses: list[torch.Tensor] = []
        max_orb = min(P.shape[1], config.shape[1])
        for b in range(P.shape[0]):
            for o in range(max_orb):
                if not (bool(orb_mask[b, o].item()) and bool(nele_mask[b, 0].item())):
                    continue
                n = int(config[b, o, 0].item())
                l = int(config[b, o, 1].item())
                if n <= l or n <= 0:
                    continue
                nodes = hydrogenic_radial_nodes(r, int(Z[b].item()), n, l)
                if nodes.numel() == 0:
                    continue
                p = P[b, o]
                amp = p.pow(2).max().detach().clamp_min(1.0e-12)
                for node in nodes:
                    width = max(0.04 * float(node.item()), 2.0e-2)
                    window = torch.exp(-0.5 * ((r - node) / width) ** 2)
                    local = (window * p.pow(2)).sum() / window.sum().clamp_min(1.0e-12)
                    losses.append(local / amp)
        if not losses:
            return P.sum() * 0.0
        return torch.stack(losses).mean()

    def _hydrogenic_node_crossing_loss(self, batch: dict[str, Any], P: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        one_electron = batch.get("nele")
        if one_electron is None:
            return P.sum() * 0.0
        config = batch["config_shells"].to(P.device)
        Z = batch["Z"].to(P.device)
        nele_mask = one_electron.to(P.device).view(-1, 1) <= 1
        r = self.model.grid.r.to(P.device).to(P.dtype)
        losses: list[torch.Tensor] = []
        max_orb = min(P.shape[1], config.shape[1])
        for b in range(P.shape[0]):
            for o in range(max_orb):
                if not (bool(orb_mask[b, o].item()) and bool(nele_mask[b, 0].item())):
                    continue
                n = int(config[b, o, 0].item())
                l = int(config[b, o, 1].item())
                if n <= l or n <= 0:
                    continue
                nodes = hydrogenic_radial_nodes(r, int(Z[b].item()), n, l)
                if nodes.numel() == 0:
                    continue
                p = P[b, o]
                amp = p.abs().max().detach().clamp_min(1.0e-12)
                for node in nodes:
                    left = (r >= 0.75 * node) & (r <= 0.95 * node)
                    right = (r >= 1.05 * node) & (r <= 1.35 * node)
                    if not bool(left.any().item() and right.any().item()):
                        continue
                    left_mean = p[left].mean()
                    right_mean = p[right].mean()
                    # Penalise same-sign means. The target is left_mean * right_mean < 0.
                    cross = torch.nn.functional.relu((left_mean * right_mean) / (amp * amp))
                    # Avoid satisfying the crossing term with nearly zero amplitude on both sides.
                    min_side_amp = 0.05 * amp
                    left_amp_penalty = torch.nn.functional.relu(min_side_amp - p[left].abs().mean()) / amp
                    right_amp_penalty = torch.nn.functional.relu(min_side_amp - p[right].abs().mean()) / amp
                    losses.append(cross + left_amp_penalty + right_amp_penalty)
        if not losses:
            return P.sum() * 0.0
        return torch.stack(losses).mean()

    def _hydrogenic_lobe_ratio_loss(self, batch: dict[str, Any], P: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        """Penalise mismatch in the lobe-wise probability share between P and P_ref.

        For every one-electron orbital with at least one radial node, partition
        the grid at the analytic nodes into ``K + 1`` lobes, compute the
        normalised fraction ``ρ_k = ∫_{I_k} P^2 dr / ∫ P^2 dr`` and its analytic
        counterpart ``ρ_k^ref``, and penalise the mean squared log-ratio over
        lobes (unweighted). This directly attacks the n-collapse mode where the
        model suppresses one lobe to deepen the energy.
        """
        one_electron = batch.get("nele")
        if one_electron is None:
            return P.sum() * 0.0
        config = batch["config_shells"].to(P.device)
        Z = batch["Z"].to(P.device)
        nele_mask = one_electron.to(P.device).view(-1, 1) <= 1
        r = self.model.grid.r.to(P.device).to(P.dtype)
        losses: list[torch.Tensor] = []
        max_orb = min(P.shape[1], config.shape[1])
        eps = 1.0e-8
        for b in range(P.shape[0]):
            for o in range(max_orb):
                if not (bool(orb_mask[b, o].item()) and bool(nele_mask[b, 0].item())):
                    continue
                n = int(config[b, o, 0].item())
                l = int(config[b, o, 1].item())
                if n <= l or n <= 0 or (n - l - 1) <= 0:
                    continue
                seg_mask = hydrogenic_lobe_partition(r, int(Z[b].item()), n, l).to(P.dtype)
                pref = hydrogenic_P_analytic(r, int(Z[b].item()), n, l)
                p = P[b, o]
                p_sq = (p * p).unsqueeze(0)  # [1, N_grid]
                pref_sq = (pref * pref).unsqueeze(0)
                p_segs = self.model.grid.integrate(p_sq * seg_mask, dim=-1)
                pref_segs = self.model.grid.integrate(pref_sq * seg_mask, dim=-1)
                p_total = p_segs.sum().clamp_min(eps)
                pref_total = pref_segs.sum().clamp_min(eps)
                p_frac = (p_segs / p_total).clamp_min(eps)
                pref_frac = (pref_segs / pref_total).clamp_min(eps)
                log_diff = torch.log(p_frac) - torch.log(pref_frac)
                losses.append(log_diff.pow(2).mean())
        if not losses:
            return P.sum() * 0.0
        return torch.stack(losses).mean()

    def _hydrogenic_sign_loss(self, batch: dict[str, Any], P: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        one_electron = batch.get("nele")
        if one_electron is None:
            return P.sum() * 0.0
        config = batch["config_shells"].to(P.device)
        Z = batch["Z"].to(P.device)
        nele_mask = one_electron.to(P.device).view(-1, 1) <= 1
        losses: list[torch.Tensor] = []
        max_orb = min(P.shape[1], config.shape[1])
        for b in range(P.shape[0]):
            for o in range(max_orb):
                if not (bool(orb_mask[b, o].item()) and bool(nele_mask[b, 0].item())):
                    continue
                n = int(config[b, o, 0].item())
                l = int(config[b, o, 1].item())
                if n <= l or n <= 0:
                    continue
                pref = hydrogenic_P_analytic(self.model.grid.r.to(P.device), int(Z[b].item()), n, l)
                p = P[b, o]
                overlap = self.model.grid.integrate(p * pref, dim=-1).detach()
                align = torch.where(overlap < 0.0, -torch.ones_like(overlap), torch.ones_like(overlap))
                p_aligned = p * align
                scale = (p_aligned.abs().max() * pref.abs().max()).detach().clamp_min(1.0e-12) * 0.05
                weight = (pref.abs() / pref.abs().max().detach().clamp_min(1.0e-12)).clamp_min(0.05)
                mismatch = 0.5 * (1.0 - torch.tanh((p_aligned * pref) / scale))
                losses.append((weight * mismatch).sum() / weight.sum().clamp_min(1.0e-12))
        if not losses:
            return P.sum() * 0.0
        return torch.stack(losses).mean()

    def _factor_amplitude_loss(self, f: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        mask = orb_mask.to(f.device).to(f.dtype).unsqueeze(-1)
        return ((f - 1.0).pow(2) * mask).sum() / (mask.sum() * f.shape[-1]).clamp_min(1.0)

    def _c_norm_loss(self, c: torch.Tensor, orb_mask: torch.Tensor) -> torch.Tensor:
        sq = c.pow(2).sum(dim=-1)
        mask = orb_mask.to(c.device).to(sq.dtype)
        return (sq * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _curriculum_state(
        epoch: int,
        n_epochs: int,
        cur_cfg: Any,
    ) -> tuple[int, float, str]:
        """ADR-018 — n-curriculum state for a given stage-1 epoch.

        Returns ``(active_n_max, lambda_prior_boost, sub_stage_name)``
        for ``epoch ∈ [0, n_epochs)`` given a ``stage1.curriculum`` cfg
        block. When the curriculum is disabled or unspecified, returns
        the "open-all-n, no boost" sentinel ``(10**6, 1.0, "off")`` so
        callers can apply the return values unconditionally without an
        ``if-curriculum-enabled`` branch on the hot path.

        Semantics (deterministic, pure function):

        * Sub-stages are laid out by ``schedule[i].fraction`` * n_epochs,
          rounded so they cover ``[0, n_epochs)`` exactly (any rounding
          slack lands in the last sub-stage). Fractions must sum to 1.0.
        * ``active_n_max`` is the sub-stage's ``n_max_active``. Orbitals
          with ``n_idx > active_n_max`` are zeroed in ``orb_mask`` at the
          entry of ``_physics_losses`` and thus contribute zero gradient
          to every loss.
        * λ_prior boost: at ``epoch == sub_start`` the boost is
          ``lambda_prior_boost``; it linearly decays to 1.0 over the
          first ``lambda_prior_boost_decay_frac`` fraction of the
          sub-stage's length, then stays at 1.0. The boost re-anchors
          the just-opened higher-n orbitals' λ residuals after the
          sub-stage's first transition.

        See ADR-018 in ``docs/design_rationale.md`` for the design
        rationale (decoupling depth vs. width gradient conflicts) and
        the consequences accepted (KAN drift on un-activated n in 1a).
        """
        OPEN_ALL = (10 ** 6, 1.0, "off")
        if cur_cfg is None or not bool(getattr(cur_cfg, "enabled", False)):
            return OPEN_ALL
        schedule = list(getattr(cur_cfg, "schedule", []) or [])
        if not schedule:
            return OPEN_ALL

        def _field(item: Any, name: str) -> Any:
            if isinstance(item, dict):
                return item[name]
            return getattr(item, name)

        boost_max = float(getattr(cur_cfg, "lambda_prior_boost", 1.0))
        decay_frac = float(
            getattr(cur_cfg, "lambda_prior_boost_decay_frac", 0.333)
        )
        n_epochs_i = max(1, int(n_epochs))

        # Lay out cumulative boundaries so they exactly tile [0, n_epochs_i).
        # The last sub-stage absorbs any rounding slack.
        boundaries: list[int] = []
        cumulative = 0
        for i, sub in enumerate(schedule):
            if i == len(schedule) - 1:
                end = n_epochs_i
            else:
                end = cumulative + int(
                    round(float(_field(sub, "fraction")) * n_epochs_i)
                )
                end = max(cumulative + 1, min(end, n_epochs_i))
            boundaries.append(end)
            cumulative = end

        epoch_i = int(epoch)
        prev_end = 0
        for i, sub in enumerate(schedule):
            sub_end = boundaries[i]
            sub_start = prev_end
            sub_len = max(1, sub_end - sub_start)
            if epoch_i < sub_end:
                n_max = int(_field(sub, "n_max_active"))
                name = str(_field(sub, "name"))
                elapsed = epoch_i - sub_start
                decay_window = max(1, int(round(decay_frac * sub_len)))
                if elapsed < decay_window:
                    t = elapsed / max(1, decay_window)
                    boost = boost_max - (boost_max - 1.0) * t
                else:
                    boost = 1.0
                return (n_max, float(boost), name)
            prev_end = sub_end

        # Past the last sub-stage's end (shouldn't happen for
        # epoch ∈ [0, n_epochs)) — return the last sub-stage's n_max
        # and no boost.
        last_n_max = int(_field(schedule[-1], "n_max_active"))
        return (last_n_max, 1.0, str(_field(schedule[-1], "name")))

    @staticmethod
    def _action_warmup_weight(
        epoch: int,
        weight_full: float,
        start_epoch: int,
        end_epoch: int,
    ) -> float:
        """Linear-ramp warmup schedule for the Bohr-Sommerfeld action weight.

        See prompts/15_loss_redesign_global_constraints.md §2.1.A.2.

        Replaces the previous ``L_PDE < action_pde_threshold`` gate, which never
        fired after ``DiracPDELoss`` was switched to ``detach_energy=True``
        (PDE residual now stabilises around 1.0–1.5 and never reaches 0.1).
        """
        if weight_full <= 0.0:
            return 0.0
        if epoch < start_epoch:
            return 0.0
        if epoch >= end_epoch:
            return float(weight_full)
        denom = max(1, int(end_epoch) - int(start_epoch))
        t = (int(epoch) - int(start_epoch)) / denom
        return float(weight_full) * float(t)

    def run_stage1(self, n_epochs: int | None = None) -> list[dict[str, float]]:
        n_epochs = int(n_epochs or self.cfg.stage1.n_epochs)
        opt = build_optimizer(self.model, self.cfg, stage=1)
        expected_steps = n_epochs * max(1, len(self.train_loader))
        try:
            cfg_total = int(self.cfg.scheduler.total_steps)
            if cfg_total < expected_steps:
                print(
                    f"[stage1][warn] scheduler.total_steps={cfg_total} < expected_steps={expected_steps}; "
                    "lr will reach 0 mid-training. Update configs/default.yaml:scheduler.total_steps to match."
                )
        except Exception:
            pass
        sched = build_scheduler(opt, self.cfg)
        ortho_cfg = getattr(self.cfg.stage1, "ortho", None)
        full_lowdin_after = int(getattr(ortho_cfg, "use_full_lowdin_after_epoch", 10**9)) if ortho_cfg is not None else 10**9
        curriculum_cfg = getattr(self.cfg.stage1, "curriculum", None)
        hist: list[dict[str, float]] = []
        prev_sub_name: str | None = None
        for epoch in range(n_epochs):
            # Toggle the model's orthonormalisation mode for the whole epoch.
            # Late stage1 switches from norm-only Löwdin to full symmetric Löwdin
            # to remove residual cross-orbital overlap (which otherwise lets
            # excited-state Rayleigh quotients drift below their true target).
            self.model.use_full_lowdin_now = bool(epoch >= full_lowdin_after)
            # ADR-018 — compute curriculum state once per epoch and stash it
            # on ``self`` so ``_physics_losses`` (called per batch) sees the
            # same value without an extra config read. ``cur_lambda_boost``
            # multiplies the ``lambda_prior`` weight only.
            cur_active_n_max, cur_lambda_boost, cur_sub_name = (
                self._curriculum_state(epoch, n_epochs, curriculum_cfg)
            )
            self._cur_active_n_max = cur_active_n_max
            if cur_sub_name != "off" and cur_sub_name != prev_sub_name:
                print(
                    f"[stage1][curriculum] epoch {epoch + 1}/{n_epochs} "
                    f"enter sub-stage '{cur_sub_name}' "
                    f"active_n_max={cur_active_n_max} "
                    f"lambda_prior_boost={cur_lambda_boost:.3f}"
                )
                prev_sub_name = cur_sub_name
            epoch_start = len(hist)
            for batch in self.train_loader:
                batch = to_device(batch, self.device)
                out = self.model(batch, use_residual=False)
                losses = self._physics_losses(batch, out)
                w = self.cfg.stage1.weights
                warmup = self.cfg.stage1.warmup
                node_target_w = float(w.node) if float(w.node) > 0.0 else float(warmup.node_weight_after_warmup)
                action_target_w = float(w.action) if float(w.action) > 0.0 else float(warmup.action_weight_after_warmup)
                shape_target_w = float(w.shape)
                shape_w = (
                    shape_target_w
                    if epoch < int(warmup.shape_pretrain_epochs)
                    else float(warmup.shape_weight_after_pretrain)
                )
                pde_w = float(w.pde)
                if epoch < int(warmup.shape_pretrain_epochs):
                    pde_w *= float(warmup.pde_weight_during_shape_pretrain)
                node_w = node_target_w if epoch >= int(warmup.node_start_epoch) else 0.0
                action_warmup_start = int(getattr(warmup, "action_warmup_start_epoch", warmup.action_start_epoch))
                action_warmup_end = int(getattr(warmup, "action_warmup_end_epoch", action_warmup_start + 70))
                action_w = self._action_warmup_weight(
                    epoch=epoch,
                    weight_full=action_target_w,
                    start_epoch=action_warmup_start,
                    end_epoch=action_warmup_end,
                )
                lobe_ratio_w = float(getattr(w, "lobe_ratio", 0.0))
                decay_w = float(getattr(w, "decay_consistency", 0.0))
                virial_w = float(getattr(w, "virial", 0.0))
                anti_collapse_w = float(getattr(w, "anti_collapse", 0.0))
                # ADR-018 — λ_prior weight is boosted at sub-stage entry to
                # re-anchor the newly-opened higher-n orbitals' λ residuals.
                lambda_prior_w = float(w.lambda_prior) * float(cur_lambda_boost)
                total = (
                    pde_w * losses["pde"]
                    + float(w.ortho) * losses["ortho"]
                    + node_w * losses["node"]
                    + float(w.node_pos) * losses["node_pos"]
                    + float(w.node_cross) * losses["node_cross"]
                    + lobe_ratio_w * losses["lobe_ratio"]
                    + float(w.sign) * losses["sign"]
                    + action_w * losses["action"]
                    + lambda_prior_w * losses["lambda_prior"]
                    + shape_w * losses["shape"]
                    + float(w.asym) * losses["asym"]
                    + float(w.smooth) * losses["smooth"]
                    + float(w.factor_amp) * losses["factor_amp"]
                    + float(w.c_norm) * losses["c_norm"]
                    + decay_w * losses["decay_consistency"]
                    + virial_w * losses["virial"]
                    + anti_collapse_w * losses["anti_collapse"]
                )
                opt.zero_grad(); total.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.optimizer.grad_clip)); opt.step(); sched.step()
                hist.append(
                    {k: float(v.detach().cpu()) for k, v in losses.items()}
                    | {
                        "total": float(total.detach().cpu()),
                        "pde_weight": pde_w,
                        "shape_weight": shape_w,
                        "action_weight": action_w,
                        "node_weight": node_w,
                        "anti_collapse_weight": anti_collapse_w,
                        # ADR-018 diagnostics
                        "lambda_prior_weight": lambda_prior_w,
                        "lambda_prior_boost": float(cur_lambda_boost),
                        "curriculum_active_n_max": int(cur_active_n_max),
                        "curriculum_sub_stage": cur_sub_name,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "epoch": epoch + 1,
                        "step": len(hist) + 1,
                    }
                )
            if len(hist) > epoch_start:
                epoch_rows = hist[epoch_start:]
                mean_total = sum(row["total"] for row in epoch_rows) / len(epoch_rows)
                mean_pde = sum(row["pde"] for row in epoch_rows) / len(epoch_rows)
                mean_ortho = sum(row["ortho"] for row in epoch_rows) / len(epoch_rows)
                mean_lam = sum(row["lambda_prior"] for row in epoch_rows) / len(epoch_rows)
                mean_shape = sum(row["shape"] for row in epoch_rows) / len(epoch_rows)
                mean_node_pos = sum(row["node_pos"] for row in epoch_rows) / len(epoch_rows)
                mean_node_cross = sum(row["node_cross"] for row in epoch_rows) / len(epoch_rows)
                mean_lobe = sum(row["lobe_ratio"] for row in epoch_rows) / len(epoch_rows)
                mean_sign = sum(row["sign"] for row in epoch_rows) / len(epoch_rows)
                mean_decay = sum(row["decay_consistency"] for row in epoch_rows) / len(epoch_rows)
                mean_virial = sum(row["virial"] for row in epoch_rows) / len(epoch_rows)
                mean_action = sum(row["action"] for row in epoch_rows) / len(epoch_rows)
                mean_node = sum(row["node"] for row in epoch_rows) / len(epoch_rows)
                mean_anti_collapse = sum(row["anti_collapse"] for row in epoch_rows) / len(epoch_rows)
                # ADR-017 diagnostics — raw (pre-normalisation) energy-dimensioned losses.
                mean_pde_raw = sum(row.get("pde_raw", 0.0) for row in epoch_rows) / len(epoch_rows)
                mean_decay_raw = sum(row.get("decay_consistency_raw", 0.0) for row in epoch_rows) / len(epoch_rows)
                mean_virial_raw = sum(row.get("virial_raw", 0.0) for row in epoch_rows) / len(epoch_rows)
                last = hist[-1]
                print(
                    f"[stage1] epoch {epoch + 1}/{n_epochs} "
                    f"sub={last.get('curriculum_sub_stage', 'off')} "
                    f"n_max={last.get('curriculum_active_n_max', 0)} "
                    f"steps={len(hist)} "
                    f"avg_total={mean_total:.6g} "
                    f"avg_pde={mean_pde:.6g} (raw={mean_pde_raw:.4g}) "
                    f"avg_ortho={mean_ortho:.6g} "
                    f"avg_lambda={mean_lam:.6g} "
                    f"avg_shape={mean_shape:.6g} "
                    f"avg_anti_collapse={mean_anti_collapse:.6g} "
                    f"avg_node_pos={mean_node_pos:.6g} "
                    f"avg_node_cross={mean_node_cross:.6g} "
                    f"avg_lobe={mean_lobe:.6g} "
                    f"avg_sign={mean_sign:.6g} "
                    f"avg_decay={mean_decay:.6g} (raw={mean_decay_raw:.4g}) "
                    f"avg_virial={mean_virial:.6g} (raw={mean_virial_raw:.4g}) "
                    f"avg_action={mean_action:.6g} "
                    f"avg_node={mean_node:.6g} "
                    f"pde_w={last['pde_weight']:.4g} "
                    f"shape_w={last['shape_weight']:.4g} "
                    f"lam_boost={last.get('lambda_prior_boost', 1.0):.3f} "
                    f"lr={last['lr']:.3g} "
                    f"last_total={last['total']:.6g}"
                )
        return hist

    def run_stage2_calibration(self, n_epochs: int | None = None) -> list[dict[str, float]]:
        if self.model.residual_head is None:
            return []
        for p in self.model.encoder.parameters(): p.requires_grad_(False)
        for p in self.model.level_encoder.parameters(): p.requires_grad_(False)
        for p in self.model.coeff_net.parameters(): p.requires_grad_(False)
        # Stage 2 keeps the orthonormalisation mode reached at the end of stage 1
        # (typically full Löwdin if cfg.stage1.ortho.use_full_lowdin_after_epoch
        # was reached). KAN main path is frozen here, so this is purely a forward
        # numerical setting; pin it to True to ensure consistent calibration.
        self.model.use_full_lowdin_now = True
        opt = build_optimizer(self.model, self.cfg, stage=2)
        n_epochs = int(n_epochs or self.cfg.stage2.n_epochs)
        sched = build_scheduler(opt, self.cfg)
        nist = NISTScalarHuberLoss(delta=float(self.cfg.losses.nist_huber_delta), align_ground=bool(self.cfg.losses.align_ground))
        hist: list[dict[str, float]] = []
        for epoch in range(n_epochs):
            epoch_start = len(hist)
            w_nist = float(self.cfg.stage2.w_nist_max) * min(1.0, (epoch + 1) / max(1, int(self.cfg.stage2.warmup_epochs)))
            for batch in self.train_loader:
                batch = to_device(batch, self.device)
                out = self.model(batch, use_residual=True)
                losses = self._physics_losses(batch, out)
                loss_nist = nist(out["macro"]["E_pred_calibrated"], batch["E_target"], batch["Z"], batch["charge"])
                total = losses["pde"] * float(self.cfg.stage1.weights.pde) + losses["action"] * float(self.cfg.stage1.weights.action) + w_nist * loss_nist
                opt.zero_grad(); total.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.optimizer.grad_clip)); opt.step(); sched.step()
                hist.append({
                    "nist": float(loss_nist.detach().cpu()),
                    "total": float(total.detach().cpu()),
                    "w_nist": w_nist,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "epoch": epoch + 1,
                    "step": len(hist) + 1,
                })
            if len(hist) > epoch_start:
                rows = hist[epoch_start:]
                mean_nist = sum(r["nist"] for r in rows) / len(rows)
                mean_total = sum(r["total"] for r in rows) / len(rows)
                last = hist[-1]
                print(
                    f"[stage2] epoch {epoch + 1}/{n_epochs} steps={len(hist)} "
                    f"avg_nist={mean_nist:.6g} avg_total={mean_total:.6g} "
                    f"w_nist={w_nist:.4g} lr={last['lr']:.3g}"
                )
        return hist

    def fit(self):
        h1 = self.run_stage1()
        h2 = self.run_stage2_calibration() if bool(self.cfg.stage2.enabled) else []
        return {"stage1": h1, "stage2": h2}
