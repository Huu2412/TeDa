"""TeDA: Text-Guided and Dialect-Aware Representation Learning for Vietnamese ASR."""

import sys
from pathlib import Path

__version__ = "0.0.2"

# Vendored ChunkFormer (ViP-VL encoder code) lives in <repo>/third_party/chunkformer and is
# imported as the top-level package `chunkformer`.
_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "chunkformer"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
