from __future__ import annotations

import math
from typing import Dict, List

import torch


def collate(batch: List[Dict], pad_multiple: int = 1) -> Dict:
    """Pad features (zeros) and tokens (0) and return lengths; texts/utt_ids stay python lists.

    pad_multiple: round the time dimension UP to the nearest multiple of this value (default 1 =
    no change). Setting pad_multiple=128 drastically reduces the number of distinct tensor shapes
    seen by cuDNN, which keeps the plan cache small and prevents host RAM from growing without
    bound. The padding is zeros and feat_lens records the TRUE lengths, so the model result is
    identical (attention / CTC masking ignores the extra frames).
    """
    B = len(batch)
    feat_lens = torch.tensor([b["feats"].size(0) for b in batch], dtype=torch.long)
    tok_lens = torch.tensor([b["tokens"].numel() for b in batch], dtype=torch.long)
    D_feat = batch[0]["feats"].size(1)
    raw_max = int(feat_lens.max())
    padded_max = math.ceil(raw_max / max(pad_multiple, 1)) * max(pad_multiple, 1)
    feats = torch.zeros(B, padded_max, D_feat, dtype=torch.float32)
    tokens = torch.zeros(B, int(tok_lens.max()), dtype=torch.long)
    for i, b in enumerate(batch):
        feats[i, : b["feats"].size(0)] = b["feats"]
        tokens[i, : b["tokens"].numel()] = b["tokens"]
    return {
        "feats": feats,
        "feat_lens": feat_lens,   # true lengths, NOT padded_max
        "tokens": tokens,
        "token_lens": tok_lens,
        "region_ids": torch.tensor([b["region_id"] for b in batch], dtype=torch.long),
        "utt_ids": [b["utt_id"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
