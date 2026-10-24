"""WER / CER (corpus level), per-region and per-group breakdowns, bootstrap confidence intervals.

Definitions (state them in the report):
  * WER = (S + D + I) / N over WHITESPACE tokens of the normalised text. Vietnamese is written with
    one syllable per token, so this is a syllable error rate. Computed at corpus level: total errors
    divided by total reference tokens (NOT the mean of per-utterance rates).
  * CER = the same on characters of the normalised string (single spaces included).
  * Reference and hypothesis are both the normalised (lower-case, no punctuation, spoken-form digits)
    text produced by teda.data.text_norm.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from teda.data.collate import collate, to_device
from teda.data.label_maps import REGIONS
from teda.data.sampler import DynamicBatchSampler
from teda.engine.decoding import ctc_greedy
from teda.utils import dist as D

log = logging.getLogger("teda")


# --------------------------------------------------------------------------- edit distance
def levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Edit distance (sub/ins/del all cost 1), row-vectorised with numpy."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    ref_a, hyp_a = np.asarray(ref), np.asarray(hyp)
    idx = np.arange(m + 1)
    prev = idx.copy()
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int64)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (hyp_a != ref_a[i - 1]), prev[1:] + 1)
        cur = np.minimum.accumulate(cur - idx) + idx  # resolves the insertion chain within the row
        prev = cur
    return int(prev[m])


def _word_ids(refs: Sequence[str], hyps: Sequence[str]):
    vocab: Dict[str, int] = {}
    enc = lambda s: [vocab.setdefault(w, len(vocab)) for w in s.split()]  # noqa: E731
    return [enc(r) for r in refs], [enc(h) for h in hyps]


def error_counts(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per-utterance error counts and reference lengths for words and characters."""
    rw, hw = _word_ids(refs, hyps)
    ew = np.array([levenshtein(a, b) for a, b in zip(rw, hw)], dtype=np.int64)
    nw = np.array([len(a) for a in rw], dtype=np.int64)
    ec = np.array([levenshtein(list(r), list(h)) for r, h in zip(refs, hyps)], dtype=np.int64)
    nc = np.array([len(r) for r in refs], dtype=np.int64)
    return {"err_w": ew, "n_w": nw, "err_c": ec, "n_c": nc}


def rate(err: np.ndarray, n: np.ndarray) -> float:
    return float(err.sum() / max(n.sum(), 1))


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_ci(err: np.ndarray, n: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI of the corpus-level rate, resampling utterances with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    r = err[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Compare two systems on the SAME utterances: delta = WER_a - WER_b (negative => a is better).
    Returns the observed delta, its 95% CI and the share of resamples where a is not better than b."""
    assert len(err_a) == len(err_b) == len(n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    den = np.maximum(n[idx].sum(1), 1)
    delta = err_a[idx].sum(1) / den - err_b[idx].sum(1) / den
    return {
        "delta": rate(err_a, n) - rate(err_b, n),
        "ci_lo": float(np.quantile(delta, 0.025)),
        "ci_hi": float(np.quantile(delta, 0.975)),
        "p_a_not_better": float((delta >= 0).mean()),
    }


# ------------------------------------------------------------------------------- summary
def summarize(records: List[Dict], n_boot: int = 1000, seed: int = 0) -> Dict:
    """records: dicts with utt_id, ref, hyp, region_id, groups (list of str)."""
    refs = [r["ref"] for r in records]
    hyps = [r["hyp"] for r in records]
    c = error_counts(refs, hyps)

    def block(mask: np.ndarray) -> Dict:
        return {
            "n_utts": int(mask.sum()),
            "wer": rate(c["err_w"][mask], c["n_w"][mask]),
            "cer": rate(c["err_c"][mask], c["n_c"][mask]),
        }

    everyone = np.ones(len(records), dtype=bool)
    out: Dict = {**block(everyone)}
    if n_boot and len(records) > 1:
        lo, hi = bootstrap_ci(c["err_w"], c["n_w"], n_boot, seed)
        out["wer_ci95"] = [lo, hi]
    rid = np.array([r["region_id"] for r in records])
    out["by_region"] = {name: block(rid == i) for i, name in enumerate(REGIONS) if (rid == i).any()}
    tags = [set(r.get("groups", [])) for r in records]
    special = np.array([bool(t) for t in tags])
    by_group = {"plain": block(~special), "special": block(special)}
    for tag in ("digit", "acronym", "foreign"):
        m = np.array([tag in t for t in tags])
        if m.any():
            by_group[tag] = block(m)
    out["by_group"] = {k: v for k, v in by_group.items() if v["n_utts"] > 0}
    out["_errors"] = {"err_w": c["err_w"].tolist(), "n_w": c["n_w"].tolist(), "utt_ids": [r["utt_id"] for r in records]}
    return out


# ------------------------------------------------------------------------------- decoding
@torch.no_grad()
def decode_dataset(model, dataset, tokenizer, ctx: D.DistContext, batch_frames: int, amp: bool = False,
                   chunk: Optional[Sequence[int]] = None, num_workers: int = 2, pin_memory: bool = False) -> List[Dict]:
    """Greedy-decode the whole dataset, split across ranks; returns records from ALL ranks (every rank)."""
    sampler = DynamicBatchSampler(dataset.num_frames, batch_frames, ctx.world_size, ctx.rank,
                                  shuffle=False, pad_to_world=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers,
                        pin_memory=pin_memory)
    was_training = model.training
    model.eval()
    recs: List[Dict] = []
    for batch in loader:
        b = to_device(batch, ctx.device)
        with torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=amp and ctx.device.type == "cuda"):
            logits, lens = model.encode_logits(b["feats"], b["feat_lens"], chunk=chunk)
        hyps = ctc_greedy(logits.float(), lens, tokenizer_blank(tokenizer))
        for i, uid in enumerate(batch["utt_ids"]):
            recs.append({
                "utt_id": uid,
                "ref": batch["texts"][i],
                "hyp": tokenizer.decode(hyps[i]),
                "region_id": int(batch["region_ids"][i]),
                "groups": sorted(dataset.groups_of(uid)),
            })
    model.train(was_training)
    merged = [r for part in D.gather_objects(recs) for r in part]
    merged.sort(key=lambda r: r["utt_id"])
    return merged


def tokenizer_blank(tokenizer) -> int:
    from teda.data.tokenizer import BLANK_ID

    return BLANK_ID


def evaluate(model, dataset, tokenizer, ctx: D.DistContext, batch_frames: int, amp: bool = False,
             chunk: Optional[Sequence[int]] = None, num_workers: int = 2, pin_memory: bool = False, n_boot: int = 1000) -> Tuple[Dict, List[Dict]]:
    """Decode + summarise. Every rank gets the same summary (computed on rank 0, then broadcast)."""
    recs = decode_dataset(model, dataset, tokenizer, ctx, batch_frames, amp, chunk, num_workers, pin_memory=pin_memory)
    summary = summarize(recs, n_boot=n_boot) if ctx.is_main else None
    summary = D.broadcast_object(summary)
    return summary, recs
