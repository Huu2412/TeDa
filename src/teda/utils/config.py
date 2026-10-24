"""YAML config handling: base.yaml + exp/<name>.yaml + command-line `--set a.b=c` overrides."""

from __future__ import annotations

import copy
import difflib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class AttrDict(dict):
    """dict with attribute access (cfg.train.lr). Nested dicts are wrapped on access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(val, dict) and not isinstance(val, AttrDict):
            val = AttrDict(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def deep_merge(base: Dict, new: Dict) -> Dict:
    """Return base updated with new. Dicts merge recursively; everything else (incl. lists) is replaced."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: Dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def path_exists(cfg: Dict, dotted: str) -> bool:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


def flat_keys(cfg: Dict, prefix: str = "") -> Iterable[str]:
    """All dotted key paths of a nested dict (intermediate nodes included)."""
    for k, v in cfg.items():
        path = f"{prefix}{k}"
        yield path
        if isinstance(v, dict):
            yield from flat_keys(v, path + ".")


def parse_override(text: str):
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)  # 5 -> int, [0,1] -> list, null -> None, true -> bool


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    base: str | Path = "configs/base.yaml",
    exp: Optional[str | Path] = None,
    overrides: Iterable[str] = (),
) -> AttrDict:
    """`exp` may be a path or a bare name (resolved to <base dir>/exp/<name>.yaml)."""
    base = Path(base)
    cfg = load_yaml(base)
    if exp:
        exp_path = Path(exp)
        if not exp_path.suffix:
            exp_path = base.parent / "exp" / f"{exp}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"experiment config not found: {exp_path}")
        exp_cfg = load_yaml(exp_path)
        cfg = deep_merge(cfg, exp_cfg)
        if "exp_name" not in exp_cfg:
            cfg["exp_name"] = exp_path.stem
    for ov in overrides:
        key, value = parse_override(ov)
        allow_new = key.startswith("+")
        key = key.lstrip("+")
        if not allow_new and not path_exists(cfg, key):
            close = difflib.get_close_matches(key, list(flat_keys(cfg)), n=3, cutoff=0.6)
            raise KeyError(
                f"override '{key}' does not exist in the config (typo?)."
                + (f" Did you mean: {', '.join(close)}?" if close else "")
                + " To add a brand-new key on purpose, write it with a leading '+', e.g. --set +a.b=1"
            )
        set_by_path(cfg, key, value)
    return AttrDict(cfg)


def to_plain(cfg: Any) -> Any:
    """Recursively convert AttrDict -> dict (safe to pickle / dump)."""
    if isinstance(cfg, dict):
        return {k: to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_plain(v) for v in cfg]
    return cfg


def dump_config(cfg: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False, allow_unicode=True)
