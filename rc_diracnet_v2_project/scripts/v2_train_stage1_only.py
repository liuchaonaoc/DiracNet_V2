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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); args = ap.parse_args()
    cfg = load_cfg(args.config); torch.manual_seed(cfg.seed)
    ds = LevelRowDataset(cfg.dataset.manifest, max_orb=cfg.dataset.max_orb, max_seq=cfg.dataset.max_seq, align_in_loader=False)
    bb = V2BatchBuilder(cfg.dataset.max_orb)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=lambda xs: bb.build(collate_levelwise(xs)))
    print(
        "[stage1] "
        f"epochs={cfg.stage1.n_epochs} "
        f"rows={len(ds)} "
        f"batch_size={cfg.training.batch_size} "
        f"batches_per_epoch={len(loader)} "
        f"expected_steps={int(cfg.stage1.n_epochs) * len(loader)}"
    )
    model = DiracNetV2(cfg, term_vocab_size=len(ds.term_vocab))
    hist = TwoStageTrainer(model, loader, loader, cfg).run_stage1(cfg.stage1.n_epochs)
    Path(cfg.training.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.log_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": to_dict(cfg)}, Path(cfg.training.ckpt_dir) / "stage1_latest.pt")
    if hist:
        history_path = Path(cfg.training.log_dir) / "stage1_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            writer.writeheader()
            writer.writerows(hist)
        print(f"[ok] wrote history {history_path}")
    print(f"[ok] stage1 steps={len(hist)} last={hist[-1] if hist else {}}")


if __name__ == "__main__": main()
