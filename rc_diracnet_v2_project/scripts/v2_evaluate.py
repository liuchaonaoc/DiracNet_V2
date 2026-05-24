from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import load_cfg
from rc_diracnet_v2.constants import HARTREE_eV
from rc_diracnet_v2.data.batch_builder import V2BatchBuilder
from rc_diracnet_v2.data.dataset import LevelRowDataset, collate_levelwise
from rc_diracnet_v2.models import DiracNetV2


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "mean_signed": float("nan")}
    mae = sum(abs(v) for v in values) / len(values)
    rmse = math.sqrt(sum(v * v for v in values) / len(values))
    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max(abs(v) for v in values),
        "mean_signed": sum(values) / len(values),
    }


def _print_summary(title: str, diffs: list[float]) -> None:
    s = _summary(diffs)
    print(title)
    print(
        f"  count={len(diffs)} "
        f"MAE={s['mae']:.3f} meV "
        f"RMSE={s['rmse']:.3f} meV "
        f"max|err|={s['max_abs']:.3f} meV "
        f"mean_signed={s['mean_signed']:.3f} meV"
    )


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--ckpt", type=Path, default=None); ap.add_argument("--manifest", type=Path, required=True); args = ap.parse_args()
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
    if args.ckpt and args.ckpt.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False); model.load_state_dict(state.get("model", state), strict=False)
    elif ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False); model.load_state_dict(state.get("model", state), strict=False)
    elif ckpt_path:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        print("[warn] no --ckpt provided; evaluating a freshly initialized model")
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch, use_residual=False)
            tgt = batch["E_target"]
            pred_main = out["macro"]["E_pred_main"].cpu()
            pred_cal = out["macro"]["E_pred_calibrated"].cpu()
            delta = out["macro"]["delta_residual"]
            if delta is None:
                delta = torch.zeros_like(pred_main)
            else:
                delta = delta.cpu()
            lam = out["lam"].cpu()
            per = out["per_orb_features"].cpu()
            config = batch["config_shells"].cpu()
            for i in range(tgt.shape[0]):
                z = int(batch["Z"][i].item())
                n = int(config[i, 0, 0].item()) if config.shape[1] > 0 else 0
                lam_ref = float(per[i, 0, 1].item() / max(per[i, 0, 2].item(), 1.0)) if per.shape[1] > 0 else float("nan")
                lam_val = float(lam[i, 0].item()) if lam.shape[1] > 0 else float("nan")
                rows.append(
                    {
                        "Z": z,
                        "n": n,
                        "target": float(tgt[i].item()),
                        "pred_main": float(pred_main[i].item()),
                        "pred_cal": float(pred_cal[i].item()),
                        "delta_mev": float(delta[i].item() * HARTREE_eV * 1000.0),
                        "err_main_mev": float((pred_main[i] - tgt[i]).item() * HARTREE_eV * 1000.0),
                        "err_cal_mev": float((pred_cal[i] - tgt[i]).item() * HARTREE_eV * 1000.0),
                        "lambda": lam_val,
                        "lambda_ref": lam_ref,
                        "lambda_rel_err": abs(lam_val - lam_ref) / max(abs(lam_ref), 1.0e-12),
                    }
                )

    diffs_main = [r["err_main_mev"] for r in rows]
    diffs_cal = [r["err_cal_mev"] for r in rows]
    print("=== E_orb-only Metrics (MAIN RESULT) ===")
    _print_summary("overall", diffs_main)
    print("=== E_orb + Δ_residual Metrics (CALIBRATION ONLY) ===")
    _print_summary("overall", diffs_cal)

    by_z: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_z[int(row["Z"])].append(row)

    print("=== By-Z Summary: E_orb-only ===")
    print("Z count MAE_meV RMSE_meV max_abs_meV mean_signed_meV mean_lambda_rel_err")
    for z in sorted(by_z):
        group = by_z[z]
        summary = _summary([r["err_main_mev"] for r in group])
        mean_lam = sum(r["lambda_rel_err"] for r in group) / len(group)
        print(
            f"{z:3d} {len(group):5d} "
            f"{summary['mae']:9.3f} {summary['rmse']:10.3f} "
            f"{summary['max_abs']:11.3f} {summary['mean_signed']:15.3f} "
            f"{mean_lam:19.6f}"
        )

    print("=== Per-Level Details: E_orb-only ===")
    print("idx Z n target_Ha pred_Ha err_meV lambda lambda_ref lambda_rel_err")
    for idx, row in enumerate(rows):
        print(
            f"{idx:03d} {int(row['Z']):3d} {int(row['n']):2d} "
            f"{row['target']: .8f} {row['pred_main']: .8f} "
            f"{row['err_main_mev']: .3f} "
            f"{row['lambda']: .6f} {row['lambda_ref']: .6f} {row['lambda_rel_err']: .6f}"
        )

if __name__ == "__main__": main()
