"""Build the offline PhoBERT word-vector cache (the logic behind scripts/build_phobert_cache.py).

Alignment units: the reference text (text_norm: spoken-form numbers, lower case) is cut into units; CIF later fires once
per unit. ``unit="word"`` uses VnCoreNLP word segmentation (PhoBERT was pretrained on segmented text: "hoc_sinh" is one
unit); ``unit="syllable"`` uses the plain space-separated syllables (no Java, PhoBERT sees unsegmented text).

A unit's vector is the mean of the last-layer hidden states of its sub-word pieces. Long texts are processed in chunks of
whole units so no unit is split across chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch


def make_segmenter(unit: str, vncorenlp_dir: str | None = None) -> Callable[[str], List[str]]:
    if unit == "syllable":
        return lambda text: text.split()
    if unit != "word":
        raise ValueError(f"unit must be 'word' or 'syllable', got {unit!r}")
    try:
        import py_vncorenlp
    except ImportError as e:  # pragma: no cover - needs Java + network
        raise SystemExit("unit=word needs VnCoreNLP: `pip install py_vncorenlp` and a JDK (apt-get install -y default-jre). "
                         "Or use --unit syllable.") from e
    save_dir = str(Path(vncorenlp_dir or "artifacts/vncorenlp").resolve())  # pragma: no cover
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # pragma: no cover
    if not (Path(save_dir) / "VnCoreNLP-1.2.jar").exists():  # pragma: no cover
        py_vncorenlp.download_model(save_dir=save_dir)
    seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)  # pragma: no cover

    def segment(text: str) -> List[str]:  # pragma: no cover
        return [w for sent in seg.word_segment(text) for w in sent.split()]

    return segment


def _chunks(piece_ids: List[List[int]], max_pieces: int) -> List[List[int]]:
    """Group unit indices into chunks whose total piece count (plus 2 specials) fits max_pieces."""
    out, cur, n = [], [], 0
    for i, p in enumerate(piece_ids):
        if cur and n + len(p) > max_pieces - 2:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(p)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def unit_vectors(units_per_text: Sequence[Sequence[str]], tokenizer, model, device, max_pieces: int = 256,
                 batch_size: int = 32) -> List[np.ndarray]:
    """For every text, a float32 array (n_units, dim): mean of the sub-word hidden states of each unit."""
    cache: Dict[str, List[int]] = {}

    def pieces(u: str) -> List[int]:
        if u not in cache:
            ids = tokenizer.encode(u, add_special_tokens=False)
            cache[u] = ids if ids else [tokenizer.unk_token_id]
        return cache[u]

    jobs = []   # (text index, [unit indices], ids)
    for ti, units in enumerate(units_per_text):
        ps = [pieces(u) for u in units]
        for chunk in _chunks(ps, max_pieces):
            ids = [tokenizer.bos_token_id] + [x for i in chunk for x in ps[i]] + [tokenizer.eos_token_id]
            jobs.append((ti, chunk, ids, [len(ps[i]) for i in chunk]))
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k][2]))
    results: List[np.ndarray | None] = [None] * len(units_per_text)
    dim = None
    parts: Dict[int, List] = {}
    for s in range(0, len(order), batch_size):
        batch = [jobs[k] for k in order[s: s + batch_size]]
        L = max(len(b[2]) for b in batch)
        ids = torch.full((len(batch), L), tokenizer.pad_token_id, dtype=torch.long)
        att = torch.zeros(len(batch), L, dtype=torch.long)
        for r, b in enumerate(batch):
            ids[r, : len(b[2])] = torch.tensor(b[2])
            att[r, : len(b[2])] = 1
        h = model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state.float().cpu()
        dim = h.size(-1)
        for r, (ti, chunk, idl, lens) in enumerate(batch):
            pos, vecs = 1, []
            for n in lens:
                vecs.append(h[r, pos: pos + n].mean(0))
                pos += n
            parts.setdefault(ti, []).append((chunk[0], torch.stack(vecs).numpy()))
    for ti, units in enumerate(units_per_text):
        if not units:
            results[ti] = np.zeros((0, dim or 0), np.float32)
        else:
            results[ti] = np.concatenate([v for _, v in sorted(parts[ti], key=lambda x: x[0])], 0)
    return results  # type: ignore[return-value]


def write_cache(out_dir: str | Path, utt_ids: Sequence[str], units: Sequence[Sequence[str]], vecs: Sequence[np.ndarray],
                train_mask: Sequence[bool], meta: Dict) -> Dict:
    """Write vectors.f16 / index.json / stats.npz / meta.json. Stats use TRAIN utterances only."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(v.shape[0] for v in vecs)
    dim = next((v.shape[1] for v in vecs if v.shape[0]), 0)
    mm = np.memmap(out / "vectors.f16", dtype=np.float16, mode="w+", shape=(total, dim))
    index, off = {}, 0
    s1 = np.zeros(dim, np.float64)
    s2 = np.zeros(dim, np.float64)
    n_tr = 0
    for uid, us, v, tr in zip(utt_ids, units, vecs, train_mask):
        n = v.shape[0]
        mm[off: off + n] = v.astype(np.float16)
        index[uid] = [off, n, " ".join(us)]
        if tr and n:
            vv = v.astype(np.float64)
            s1 += vv.sum(0)
            s2 += (vv ** 2).sum(0)
            n_tr += n
        off += n
    mm.flush()
    mean = s1 / max(n_tr, 1)
    std = np.sqrt(np.maximum(s2 / max(n_tr, 1) - mean ** 2, 1e-8))
    np.savez(out / "stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    meta = {**meta, "dim": int(dim), "n_vectors": int(total), "n_utts": len(index), "n_train_words": int(n_tr)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
