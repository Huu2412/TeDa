"""Accent / dialect label map: three regions only (North, Central, South)."""

from __future__ import annotations

import unicodedata

REGIONS = ("North", "Central", "South")

_ALIASES = {
    "north": 0, "bac": 0, "bắc": 0, "mien bac": 0, "miền bắc": 0,
    "central": 1, "trung": 1, "mien trung": 1, "miền trung": 1,
    "south": 2, "nam": 2, "mien nam": 2, "miền nam": 2,
}


def region_to_id(value: str) -> int:
    """Map a ViMD `region` string to 0/1/2. Raises ValueError on unknown values."""
    key = unicodedata.normalize("NFC", str(value)).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown region value {value!r}. Known: {sorted(_ALIASES)}. "
            "Add the new spelling to _ALIASES in label_maps.py."
        )
    return _ALIASES[key]


def id_to_region(idx: int) -> str:
    return REGIONS[idx]
