from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import _PROJECT_ROOT
from rc_diracnet_v2.constants import HARTREE_eV


def build(n_levels: int = 6, z_max: int = 3):
    rows = []
    for Z in range(1, z_max + 1):
        q = Z - 1
        for n in range(1, n_levels + 1):
            E = -(Z * Z) / (2.0 * n * n)
            cfg = f"{n}s1"
            rows.append({"Z": Z, "ion_charge": q, "parent_config": cfg, "level_config": cfg, "J": 0.5, "parity": 0, "level_eV": E * HARTREE_eV, "uncertainty_eV": 0.0, "term": "^2S_{1/2}"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_PROJECT_ROOT / "data_cache/manifest_hydrogenic_v2.parquet")
    ap.add_argument("--n-levels", type=int, default=6)
    ap.add_argument("--z-max", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.n_levels, args.z_max).to_parquet(args.out)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
