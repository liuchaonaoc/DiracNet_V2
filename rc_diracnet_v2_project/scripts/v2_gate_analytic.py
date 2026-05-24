from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import load_cfg
from rc_diracnet_v2.data.batch_builder import V2BatchBuilder
from rc_diracnet_v2.data.dataset import LevelRowDataset, collate_levelwise
from rc_diracnet_v2.models import DiracNetV2
from rc_diracnet_v2.training.stage_gate import check_stage1_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/v2_phase1_stage1_pde_only.yaml"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    ds = LevelRowDataset(args.manifest, max_orb=cfg.dataset.max_orb, max_seq=cfg.dataset.max_seq, align_in_loader=False)
    bb = V2BatchBuilder(cfg.dataset.max_orb)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, collate_fn=lambda xs: bb.build(collate_levelwise(xs)))
    model = DiracNetV2(cfg, term_vocab_size=len(ds.term_vocab))
    if args.ckpt and args.ckpt.exists():
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    rep = check_stage1_gate(model, loader, cfg.stage1.gate)
    print(rep.to_markdown())


if __name__ == "__main__":
    main()
