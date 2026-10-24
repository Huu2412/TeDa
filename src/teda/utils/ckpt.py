"""Checkpoint helpers: atomic saves, last/best bookkeeping, resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def atomic_save(obj: Any, path: str | Path) -> None:
    """Write to a temp file then rename, so an interrupted session never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str | Path, map_location="cpu") -> Dict[str, Any]:
    # Our own checkpoints hold optimizer / scheduler state objects, hence weights_only=False.
    # Only load files you created yourself.
    return torch.load(path, map_location=map_location, weights_only=False)


def write_json(path: str | Path, obj: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
