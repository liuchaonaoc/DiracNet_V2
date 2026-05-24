"""Tiny config wrapper compatible with the subset of OmegaConf we use.

兼容性目的
----------
* 优先使用 ``omegaconf.OmegaConf``；
* 若环境未安装，则使用此处的 ``DotDict`` 仿真：
    - 支持 ``cfg.foo.bar`` 属性访问；
    - 支持 ``cfg.get("foo", default)`` 和 ``cfg["foo"]``；
    - 支持 ``dict(cfg)`` / ``OmegaConf.create(dict)`` 风格构造。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf  # noqa: F401
    _HAS_OMEGACONF = True
except ImportError:                                # pragma: no cover
    _HAS_OMEGACONF = False


class DotDict(dict):
    """支持属性式访问的嵌套字典。"""

    def __init__(self, data: dict | None = None) -> None:
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = _wrap(v)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


def _wrap(v: Any) -> Any:
    if isinstance(v, dict) and not isinstance(v, DotDict):
        return DotDict(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def create(data: dict) -> Any:
    """OmegaConf-like create()."""
    if _HAS_OMEGACONF:
        return OmegaConf.create(data)
    return DotDict(data)


def load(path: str | Path) -> Any:
    """OmegaConf-like load()."""
    if _HAS_OMEGACONF:
        return OmegaConf.load(str(path))
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DotDict(data)


def merge(*configs) -> Any:
    if _HAS_OMEGACONF:
        return OmegaConf.merge(*configs)
    out: dict[str, Any] = {}
    for c in configs:
        for k, v in dict(c).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = dict(merge(out[k], v))
            else:
                out[k] = v
    return DotDict(out)


def to_dict(cfg: Any) -> dict:
    """OmegaConf-like to_container."""
    if _HAS_OMEGACONF and "OmegaConf" in globals():
        try:
            from omegaconf import OmegaConf as _OC
            return _OC.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
        except Exception:
            pass
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [to_dict(x) for x in cfg]
    return cfg
