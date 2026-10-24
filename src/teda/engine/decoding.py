from __future__ import annotations

from typing import List

import torch


def ctc_greedy(logits: torch.Tensor, lens: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """Best-path CTC decoding: argmax per frame, collapse repeats, drop blanks. logits: (B, T, V)."""
    ids = logits.argmax(dim=-1).cpu()
    out: List[List[int]] = []
    for b in range(ids.size(0)):
        seq = ids[b, : int(lens[b])].tolist()
        prev, res = -1, []
        for t in seq:
            if t != prev and t != blank_id:
                res.append(t)
            prev = t
        out.append(res)
    return out
