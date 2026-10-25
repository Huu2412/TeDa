"""Manifest-backed dataset: wav16k + normalised text -> fbank features, BPE ids, region label."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from teda.data import features as F
from teda.data.groups import utt_groups
from teda.data.label_maps import region_to_id
from teda.data.tokenizer import BPETokenizer
from teda.utils.manifest import read_tsv

log = logging.getLogger("teda")


def ctc_required_frames(tokens: List[int]) -> int:
    """Minimum number of encoder frames CTC needs: len(tokens) + number of adjacent repeated tokens."""
    return len(tokens) + sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)


class VimdDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        manifest: str,
        split: str,
        tokenizer: BPETokenizer,
        feat_cfg: Dict,
        train: bool,
        text_field: str = "text_norm",
        max_dur: Optional[float] = None,
        drop_infeasible_ctc: bool = False,
        limit: Optional[int] = None,
    ):
        self.root = Path(data_dir)
        self.split = split
        self.train = train
        self.feat_cfg = feat_cfg
        self.num_mel_bins = int(feat_cfg.get("num_mel_bins", 80))
        self.text_field = text_field

        rows = [r for r in read_tsv(self.root / manifest) if r["split"] == split]
        if not rows:
            raise ValueError(f"no rows for split '{split}' in {self.root / manifest}")
        if limit:
            rows = rows[:limit]

        kept: List[Dict] = []
        n_long = n_infeasible = 0
        for r in rows:
            dur = float(r["duration"])
            toks = tokenizer.encode(r[text_field])
            if train:
                if max_dur and dur > max_dur:
                    n_long += 1
                    continue
                if drop_infeasible_ctc:
                    t_out = F.subsampled_length(F.num_fbank_frames(int(round(dur * F.SAMPLE_RATE))))
                    if ctc_required_frames(toks) > t_out:
                        n_infeasible += 1
                        continue
            if not toks:
                raise ValueError(f"{r['utt_id']}: empty target after tokenisation")
            r["_tokens"] = toks
            r["_region_id"] = region_to_id(r["region"])
            r["_dur"] = dur
            kept.append(r)
        self.rows = kept
        self._index = {r["utt_id"]: i for i, r in enumerate(kept)}
        # Dropping is ONLY ever applied to the train split; valid/test always keep every utterance.
        if train and (n_long or n_infeasible):
            log.info("[%s] dropped for training: %d too long (> %s s), %d CTC-infeasible (tokens > frames); kept %d/%d",
                     split, n_long, max_dur, n_infeasible, len(kept), len(rows))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def durations(self) -> List[float]:
        return [r["_dur"] for r in self.rows]

    @property
    def num_frames(self) -> List[int]:
        return [F.num_fbank_frames(int(round(r["_dur"] * F.SAMPLE_RATE))) for r in self.rows]

    def groups_of(self, utt_id: str):
        r = self.rows[self._index[utt_id]]
        return utt_groups(r["text_raw"], r[self.text_field])

    def __getitem__(self, i: int) -> Dict:
        r = self.rows[i]
        data, sr = sf.read(self.root / r["wav"], dtype="float32", always_2d=False)
        if sr != F.SAMPLE_RATE:
            raise ValueError(f"{r['wav']}: expected 16 kHz, got {sr} (run scripts/prepare_data.py)")
        wave = torch.from_numpy(np.ascontiguousarray(data))
        if self.train:
            wave = F.speed_perturb(wave, self.feat_cfg.get("speed_perturb") or [])
        dither = self.feat_cfg.get("dither_train", 1.0) if self.train else self.feat_cfg.get("dither_eval", 0.0)
        feats = F.compute_fbank(wave, self.num_mel_bins, dither=dither)
        sa = self.feat_cfg.get("spec_aug") or {}
        if self.train and sa.get("enabled", False):
            feats = F.spec_aug(feats, sa.get("num_t_mask", 2), sa.get("num_f_mask", 2), sa.get("max_t", 50), sa.get("max_f", 10))
        return {
            "utt_id": r["utt_id"],
            "feats": feats,
            "tokens": torch.tensor(r["_tokens"], dtype=torch.long),
            "region_id": r["_region_id"],
            "text": r[self.text_field],
        }
