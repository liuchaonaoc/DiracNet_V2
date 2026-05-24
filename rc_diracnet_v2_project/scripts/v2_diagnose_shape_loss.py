"""Diagnose why ``L_shape`` plateaus at ~0.43 after 1000 epoch.

For each level (Z, n, l) in the manifest, this script computes:

* the **same-(n,l) cosine²** that ``L_shape`` actually optimises;
* a **scan of cosine²** against analytic references with the same ``l`` but
  different ``n′ ∈ {1, …, n+2}`` — if cos² with a *lower-n* reference is
  non-trivial, the predicted ψ is contaminated by lower-n states (the
  hypothesised "1s leakage into n=2");
* the **numerical zero count** of P(r), compared to the analytic node count
  ``n − l − 1``;
* the **λ** vs ``Z/n`` deviation (already covered by the evaluator, repeated
  here for context);
* the **L² shape error** ``‖P̂_pred − sign·P̂_an‖ / ‖P̂_an‖`` (with sign
  chosen to minimise — invariant to overall phase);
* a **dump of normalised P_pred and P_an on the radial grid** to a CSV for
  off-line plotting (one file per ion).

The output is meant to disambiguate three hypotheses for why ``L_shape``
won't go below ~0.43:

  H1. Capacity bottleneck — model cannot represent the analytic shape.
      ⇒ the predicted P is *smooth* but *systematically off* (e.g. wrong
        amplitude/scale that B-splines + envelope can't reach).
  H2. Wrong-n contamination — ψ_pred(n=2) bleeds into ψ_an(n=1) territory.
      ⇒ cos²(ψ_pred(n=2), ψ_an(n=1)) noticeably > 0 AND wrong node count.
  H3. Optimisation stagnation — model is in a flat region of L_shape.
      ⇒ would manifest as very small ∂L_shape/∂c gradients (separate
        check, not in this script).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import load_cfg
from rc_diracnet_v2.constants import HARTREE_eV
from rc_diracnet_v2.data.batch_builder import V2BatchBuilder
from rc_diracnet_v2.data.dataset import LevelRowDataset, collate_levelwise
from rc_diracnet_v2.models import DiracNetV2
from rc_diracnet_v2.physics.hydrogenic_analytic import (
    hydrogenic_P_analytic,
    hydrogenic_radial_nodes,
)


def _norm_l2(P: torch.Tensor, grid) -> torch.Tensor:
    return grid.integrate(P * P, dim=-1).clamp_min(1.0e-30).sqrt()


def _cos2(P_a: torch.Tensor, P_b: torch.Tensor, grid) -> float:
    """Squared cosine between two radial functions on the grid."""
    num = grid.integrate(P_a * P_b, dim=-1)
    den = _norm_l2(P_a, grid) * _norm_l2(P_b, grid)
    return float((num / den.clamp_min(1.0e-30)).pow(2).item())


def _signed_cos(P_a: torch.Tensor, P_b: torch.Tensor, grid) -> float:
    num = grid.integrate(P_a * P_b, dim=-1)
    den = _norm_l2(P_a, grid) * _norm_l2(P_b, grid)
    return float((num / den.clamp_min(1.0e-30)).item())


def _l2_shape_error(P_pred: torch.Tensor, P_ref: torch.Tensor, grid) -> float:
    """Sign-invariant L² shape error ‖P̂_pred − s·P̂_ref‖ / ‖P̂_ref‖, s = ±1."""
    np_n = _norm_l2(P_pred, grid).clamp_min(1.0e-30)
    nr_n = _norm_l2(P_ref, grid).clamp_min(1.0e-30)
    P_pred_n = P_pred / np_n
    P_ref_n = P_ref / nr_n
    # try both signs, take the smaller
    diff_plus = grid.integrate((P_pred_n - P_ref_n).pow(2), dim=-1).clamp_min(0.0).sqrt()
    diff_minus = grid.integrate((P_pred_n + P_ref_n).pow(2), dim=-1).clamp_min(0.0).sqrt()
    return float(torch.minimum(diff_plus, diff_minus).item())


def _count_sign_changes(P: torch.Tensor, grid, r_min_skip: float = 0.05) -> int:
    """Count sign changes in P(r), ignoring the inner ``r < r_min_skip``
    region to avoid spurious zeros from numerical noise near origin."""
    r = grid.r
    mask = r > r_min_skip
    Pm = P[mask]
    if Pm.numel() < 2:
        return 0
    s = torch.sign(Pm)
    # ignore zeros (no sign change)
    s = s[s != 0]
    if s.numel() < 2:
        return 0
    return int((s[1:] != s[:-1]).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument(
        "--dump-curves",
        type=Path,
        default=None,
        help="Optional directory to dump (P_pred, P_an) curves as CSV.",
    )
    ap.add_argument(
        "--n-scan-max",
        type=int,
        default=2,
        help="Scan cos² against analytic ψ for n' in [max(l+1,1), n + N_SCAN_MAX].",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ds = LevelRowDataset(args.manifest, max_orb=cfg.dataset.max_orb, max_seq=cfg.dataset.max_seq, align_in_loader=False)
    bb = V2BatchBuilder(cfg.dataset.max_orb)
    loader = DataLoader(ds, batch_size=max(1, len(ds)), collate_fn=lambda xs: bb.build(collate_levelwise(xs)))

    model = DiracNetV2(cfg, term_vocab_size=len(ds.term_vocab))
    ckpt_path = args.ckpt
    if ckpt_path is None:
        default_ckpt = Path(cfg.training.ckpt_dir) / "stage1_latest.pt"
        if default_ckpt.exists():
            ckpt_path = default_ckpt
            print(f"[info] using default checkpoint: {ckpt_path}")
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
    else:
        print("[warn] no checkpoint found; diagnosing randomly-initialised model")
    model.eval()

    grid = model.grid
    r_np = grid.r.detach().cpu().numpy()

    if args.dump_curves is not None:
        args.dump_curves.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("D.2 / D.4  cos² scan: ψ_pred vs ψ_an(n', l) for several n'")
    print("=" * 96)

    # Column header
    n_scan_max = int(args.n_scan_max)
    header = f"{'idx':>3} {'Z':>2} {'n':>2} {'l':>2} {'λ_pred':>8} {'λ_an':>8}  {'cos²(same)':>11} {'shape_loss':>10} {'nodes(P)':>8}/{'nodes_an':>8} {'L2_err':>7}  cos²(n′)"
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch, use_residual=False)
            P = out["wavefunctions"]["P"].cpu()
            lam_pred = out["lam"].cpu()
            E_pred = out["macro"]["E_pred_main"].cpu()
            tgt = batch["E_target"].cpu()
            config = batch["config_shells"].cpu()
            Z_b = batch["Z"].cpu()
            for i in range(P.shape[0]):
                Z = int(Z_b[i].item())
                n = int(config[i, 0, 0].item())
                l = int(config[i, 0, 1].item())
                lam_p = float(lam_pred[i, 0].item())
                lam_a = float(Z) / float(n)

                P_pred = P[i, 0]  # single active orbital (hydrogenic dataset)
                # analytic same-(n, l) reference
                P_ref_same = hydrogenic_P_analytic(grid.r, Z, n, l)
                cos2_same = _cos2(P_pred, P_ref_same, grid)
                shape_loss = 1.0 - cos2_same
                l2_err = _l2_shape_error(P_pred, P_ref_same, grid)
                signed_same = _signed_cos(P_pred, P_ref_same, grid)

                # cos² against ψ_an(n', l) for n' in [l+1, n+n_scan_max]
                cos2_scan: dict[int, float] = {}
                signed_scan: dict[int, float] = {}
                for n_prime in range(max(l + 1, 1), n + n_scan_max + 1):
                    if n_prime <= l:
                        continue
                    P_ref_np = hydrogenic_P_analytic(grid.r, Z, n_prime, l)
                    cos2_scan[n_prime] = _cos2(P_pred, P_ref_np, grid)
                    signed_scan[n_prime] = _signed_cos(P_pred, P_ref_np, grid)

                # node counting
                nodes_pred = _count_sign_changes(P_pred, grid)
                nodes_an = n - l - 1

                # error in meV (for cross-check vs evaluator)
                err_meV = (float(E_pred[i].item()) - float(tgt[i].item())) * HARTREE_eV * 1000.0

                scan_str = "  ".join(
                    f"n′={k}:{v:.3f}{'*' if k == n else ''}"
                    for k, v in sorted(cos2_scan.items())
                )

                print(
                    f"{i:3d} {Z:2d} {n:2d} {l:2d} "
                    f"{lam_p:8.4f} {lam_a:8.4f}  "
                    f"{cos2_same:11.4f} {shape_loss:10.4f} "
                    f"{nodes_pred:8d}/{nodes_an:8d} "
                    f"{l2_err:7.4f}  {scan_str}"
                )

                results.append(
                    {
                        "Z": Z, "n": n, "l": l,
                        "lam_pred": lam_p, "lam_an": lam_a,
                        "cos2_same": cos2_same, "shape_loss": shape_loss,
                        "signed_same": signed_same,
                        "cos2_scan": cos2_scan, "signed_scan": signed_scan,
                        "nodes_pred": nodes_pred, "nodes_an": nodes_an,
                        "l2_err": l2_err, "err_meV": err_meV,
                    }
                )

                if args.dump_curves is not None:
                    # write a CSV: r, P_pred (normalised), P_an_same (normalised),
                    # then P_an_(n=1..n+1) (normalised) for the same l
                    import csv
                    np_norm = _norm_l2(P_pred, grid).clamp_min(1.0e-30).item()
                    fname = f"Z{Z}_n{n}_l{l}.csv"
                    fpath = args.dump_curves / fname
                    refs: list[tuple[str, torch.Tensor]] = [
                        (f"P_an_n{n}l{l}", hydrogenic_P_analytic(grid.r, Z, n, l)),
                    ]
                    for n_prime in sorted(cos2_scan.keys()):
                        refs.append((f"P_an_n{n_prime}l{l}", hydrogenic_P_analytic(grid.r, Z, n_prime, l)))
                    cols = {"r": r_np, "P_pred_normed": (P_pred / np_norm).detach().cpu().numpy()}
                    for name, R in refs:
                        rn = _norm_l2(R, grid).clamp_min(1.0e-30).item()
                        cols[name + "_normed"] = (R / rn).detach().cpu().numpy()
                    with fpath.open("w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(list(cols.keys()))
                        for k in range(len(r_np)):
                            writer.writerow([cols[c][k] for c in cols.keys()])

    print()
    print("=" * 96)
    print("Summary by (Z, n, l)")
    print("=" * 96)
    n_total = len(results)
    n_node_correct = sum(1 for r in results if r["nodes_pred"] == r["nodes_an"])
    mean_cos2 = sum(r["cos2_same"] for r in results) / max(n_total, 1)
    mean_shape = sum(r["shape_loss"] for r in results) / max(n_total, 1)
    mean_l2 = sum(r["l2_err"] for r in results) / max(n_total, 1)

    print(
        f"levels={n_total}  mean cos²(same)={mean_cos2:.4f}  "
        f"mean shape_loss={mean_shape:.4f}  mean L²err={mean_l2:.4f}  "
        f"correct node count: {n_node_correct}/{n_total}"
    )

    # Highlight wrong-n contamination
    print()
    print("Wrong-n contamination check (cos²(ψ_pred, ψ_an(n′, l)) ≥ 0.1 for some n′ ≠ n):")
    contaminated = 0
    for r in results:
        bad: list[str] = []
        for n_prime, c2 in r["cos2_scan"].items():
            if n_prime != r["n"] and c2 >= 0.1:
                bad.append(f"n′={n_prime}: cos²={c2:.3f}")
        if bad:
            contaminated += 1
            print(f"  (Z={r['Z']}, n={r['n']}, l={r['l']}): " + ", ".join(bad))
    if contaminated == 0:
        print("  (none — predicted ψ is locally orthogonal to other-n analytic refs)")
    print(f"  → {contaminated}/{n_total} levels show wrong-n leakage")


if __name__ == "__main__":
    main()
