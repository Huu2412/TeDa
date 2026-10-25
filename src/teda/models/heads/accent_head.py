from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class AccentHead(nn.Module):
    """Accent / Dialect classification head on top of ChunkFormer encoder representations.

    Pools time frames using masked average pooling, applies LayerNorm + Dropout, and
    projects to `num_classes` (default 3: North, Central, South).
    Loss is Cross-Entropy computed in fp32 for stability under amp fp16.
    """

    def __init__(self, dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, num_classes)

    def logits(self, enc: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        # enc: (B, T, D), lens: (B,)
        mask = torch.arange(enc.size(1), device=enc.device)[None, :] < lens[:, None]  # (B, T)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (enc * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        return self.proj(self.drop(self.norm(pooled)))  # (B, num_classes)

    def loss_and_acc(self, enc: torch.Tensor, lens: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(enc, lens)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            loss = F.cross_entropy(logits.float(), targets.long())
            pred = logits.argmax(dim=-1)
            acc = (pred == targets).float().mean()
        return loss, acc
