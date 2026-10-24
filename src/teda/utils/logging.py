"""Logging: human-readable log on rank 0 only + a JSONL metrics file."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

LOGGER_NAME = "teda"


def setup_logging(rank: int = 0, log_file: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [rank%(rank)d] %(message)s", "%H:%M:%S")

    class _Rank(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.addFilter(_Rank())
    logger.addHandler(sh)
    if log_file is not None and rank == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(_Rank())
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


class MetricsWriter:
    """Append-only JSONL. Only rank 0 writes."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields) -> None:
        if not self.enabled:
            return
        fields.setdefault("time", round(time.time(), 1))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")
