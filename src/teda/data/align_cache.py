"""Offline PhoBERT word-vector cache (built by scripts/build_phobert_cache.py, read here without PhoBERT).

Layout of the cache directory:
    meta.json      {"dim": 768, "n_vectors": N, "unit": "word"|"syllable", "model": ..., "text_field": ...}
    index.json     {utt_id: [offset, n_words, "w1 w2 ..."]}   words are the alignment units (CIF fires once per word)
    vectors.f16    float16 (n_vectors, dim), raw PhoBERT last-layer word vectors, memory-mapped
    stats.npz      mean, std per dimension, computed on TRAIN words only

Targets are standardised with these stats: PhoBERT vectors share a large common component, and without removing it
cosine similarity between ANY two words is already high, which would make the alignment loss trivially small.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


class AlignCache:
    def __init__(self, directory: str | Path, mode: str = "phobert"):
        if mode not in ("phobert", "random_word", "shuffle"):
            raise ValueError(f"unknown target mode {mode!r}")
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"{d}/meta.json not found: build it with scripts/build_phobert_cache.py")
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.dim = int(self.meta["dim"])
        self.index = json.loads((d / "index.json").read_text(encoding="utf-8"))
        st = np.load(d / "stats.npz")
        self.mean = st["mean"].astype(np.float32)
        self.std = np.maximum(st["std"].astype(np.float32), 1e-4)
        self.mode = mode
        self._vec: Optional[np.memmap] = None   # opened lazily so each DataLoader worker maps the file itself

    def __contains__(self, utt_id: str) -> bool:
        return utt_id in self.index

    def n_words(self, utt_id: str) -> int:
        e = self.index.get(utt_id)
        return int(e[1]) if e else 0

    def _vectors(self) -> np.memmap:
        if self._vec is None:
            self._vec = np.memmap(self.dir / "vectors.f16", dtype=np.float16, mode="r",
                                  shape=(int(self.meta["n_vectors"]), self.dim))
        return self._vec

    @staticmethod
    def random_vector(word: str, dim: int) -> np.ndarray:
        """Control targets: a fixed random vector per word identity (same word -> same vector), no PhoBERT semantics."""
        rng = np.random.default_rng(zlib.crc32(word.encode("utf-8")))
        return rng.standard_normal(dim).astype(np.float32)

    def get(self, utt_id: str) -> Optional[Tuple[List[str], torch.Tensor]]:
        e = self.index.get(utt_id)
        if e is None:
            return None
        off, n, text = int(e[0]), int(e[1]), e[2]
        words = text.split(" ") if n else []
        if self.mode == "random_word":
            v = np.stack([self.random_vector(w, self.dim) for w in words]) if n else np.zeros((0, self.dim), np.float32)
        else:
            v = (np.asarray(self._vectors()[off: off + n], dtype=np.float32) - self.mean) / self.std
        return words, torch.from_numpy(np.ascontiguousarray(v)).to(torch.float16)
