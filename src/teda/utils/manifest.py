"""Tiny TSV manifest helpers (no pandas dependency).

Text is sanitised (tabs / newlines -> space) before writing, so plain
tab-separated values without quoting are safe.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Written by prepare_data.py (audio + raw metadata). `wav` is relative to the
# dataset root so the folder can be moved / mounted anywhere (e.g. Kaggle).
RAW_FIELDS = [
    "utt_id", "split", "wav", "duration", "orig_sr",
    "text_raw", "region", "province_code", "province_name", "gender",
]

# Written by scan_text.py (adds normalised text and labels).
FULL_FIELDS = RAW_FIELDS + [
    "region_id", "text_clean", "text_norm", "has_digit", "norm_changed",
]


def sanitize(value) -> str:
    s = "" if value is None else str(value)
    return " ".join(s.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def write_tsv(path: str | Path, rows: Iterable[Dict], fields: Sequence[str]) -> int:
    """Atomically write rows to `path`. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([sanitize(r.get(k, "")) for k in fields])
            n += 1
    os.replace(tmp, path)
    return n


def read_tsv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        return [dict(row) for row in r]
