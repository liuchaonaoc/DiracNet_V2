"""Stage 2 calibration entrypoint: continue from a stage1 checkpoint.

Loads a stage1 checkpoint into a fresh DiracNetV2 and runs the
NIST-residual calibration loop (frozen KAN main + tanh-bounded
residual head). Saves a stage2 checkpoint and history CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import load_cfg
from rc_diracnet_v2.data.batch_builder import V2BatchBuilder
from rc_diracnet_v2.data.dataset import LevelRowDataset, collate_levelwise
from rc_diracnet_v2.models import DiracNetV2
from rc_diracnet_v2.training.two_stage_trainer import TwoStageTrainer
from rc_diracnet_v2.utils.config import to_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--stage1-ckpt",
        type=Path,
        default=None,
        help="Path to stage1 checkpoint. Defaults to <cfg.training.ckpt_dir>/stage1_latest.pt",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)

    ds = LevelRowDataset(
        cfg.dataset.manifest,
        max_orb=cfg.dataset.max_orb,
        max_seq=cfg.dataset.max_seq,
        align_in_loader=False,
    )
    bb = V2BatchBuilder(cfg.dataset.max_orb)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda xs: bb.build(collate_levelwise(xs)),
    )

    model = DiracNetV2(cfg, term_vocab_size=len(ds.term_vocab))

    ckpt_path = args.stage1_ckpt or (Path(cfg.training.ckpt_dir) / "stage1_latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"stage1 checkpoint not found: {ckpt_path}. "
            "Run scripts/v2_train_stage1_only.py first."
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(state["model"], strict=False)
    print(f"[stage2] loaded stage1 ckpt: {ckpt_path}")
    if getattr(msg, "missing_keys", None):
        print(f"[stage2] missing keys: {msg.missing_keys[:10]}{'...' if len(msg.missing_keys) > 10 else ''}")
    if getattr(msg, "unexpected_keys", None):
        print(f"[stage2] unexpected keys: {msg.unexpected_keys[:10]}{'...' if len(msg.unexpected_keys) > 10 else ''}")

    if model.residual_head is None:
        raise RuntimeError(
            "model.residual_head is None — set stage2.enabled: true in the config "
            "so DiracNetV2 instantiates the LevelResidualHead."
        )

    print(
        "[stage2] "
        f"epochs={cfg.stage2.n_epochs} "
        f"rows={len(ds)} "
        f"batch_size={cfg.training.batch_size} "
        f"batches_per_epoch={len(loader)} "
        f"expected_steps={int(cfg.stage2.n_epochs) * len(loader)} "
        f"w_nist_max={cfg.stage2.w_nist_max} "
        f"warmup_epochs={cfg.stage2.warmup_epochs} "
        f"delta_max_meV={cfg.stage2.delta_max_meV}"
    )

    trainer = TwoStageTrainer(model, loader, loader, cfg)
    hist = trainer.run_stage2_calibration(int(cfg.stage2.n_epochs))

    Path(cfg.training.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.log_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cfg": to_dict(cfg)},
        Path(cfg.training.ckpt_dir) / "stage2_latest.pt",
    )
    if hist:
        history_path = Path(cfg.training.log_dir) / "stage2_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            writer.writeheader()
            writer.writerows(hist)
        print(f"[ok] wrote history {history_path}")
    print(f"[ok] stage2 steps={len(hist)} last={hist[-1] if hist else {}}")


if __name__ == "__main__":
    main()
