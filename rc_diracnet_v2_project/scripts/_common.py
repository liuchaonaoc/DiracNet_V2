from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from rc_diracnet_v2.utils.config import create


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_cfg_data(path: Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and "defaults" in data:
        base = _load_cfg_data(path.parent / data.pop("defaults"))
        return _deep_merge(base, data)
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping: {path}")
    return data


def load_cfg(path: Path):
    data = _load_cfg_data(path)
    return create(data)
