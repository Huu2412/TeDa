from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CTCHead(nn.Module):
    """Linear projection to BPE units + CTC loss (computed in fp32, safe under fp16 autocast)."""

    def __init__(self, dim: int, vocab_size: int, blank_id: int = 0, dropout: float = 0.0):
        super().__init__()
        self.blank_id = blank_id
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, vocab_size)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(h))

    def loss(self, h: torch.Tensor, h_lens: torch.Tensor, tokens: torch.Tensor, token_lens: torch.Tensor) -> torch.Tensor:
        """Sum of per-utterance CTC losses divided by the batch size (utterances)."""
        logits = self.logits(h)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logp = torch.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # (T, B, V)
            keep = torch.arange(tokens.size(1), device=tokens.device)[None, :] < token_lens[:, None]
            targets = tokens[keep]  # concatenated targets
            loss = F.ctc_loss(
                logp, targets, h_lens.long(), token_lens.long(),
                blank=self.blank_id, reduction="sum", zero_infinity=True,
            )
        return loss / h.size(0)
