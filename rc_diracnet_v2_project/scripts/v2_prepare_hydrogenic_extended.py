from v2_prepare_hydrogenic import build
from pathlib import Path
from _common import _PROJECT_ROOT

out = _PROJECT_ROOT / "data_cache/manifest_hydrogenic_v2_extended.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
build(n_levels=6, z_max=10).to_parquet(out)
print(f"[ok] wrote {out}")
