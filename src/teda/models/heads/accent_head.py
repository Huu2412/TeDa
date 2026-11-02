"""Accent head (E01): predicts the speaker's region (North / Central / South) from the encoder.

    12 block outputs --(per-layer LayerNorm, softmax-weighted sum)--> (B, T', D)
                     --(masked attentive pooling)------------------> (B, D)   [or mean+std: pooling="asp"]
                     --(LayerNorm, Linear, GELU, Dropout, Linear)--> 3 logits

Why a weighted sum over blocks: the last block is pulled towards phonetic content by the CTC loss, while
regional pronunciation cues are usually clearer in the middle blocks. The mixing weights are learned
(softmax over 12 numbers, initialised uniform), so after training they show WHICH blocks carry accent
information, which is itself a result worth reporting. layers="last" uses only the final output (ablation).

Training-time only: this module is not used at inference, so ASR cost is unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class AccentHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_classes: int = 3,
        layers: str = "weighted",
        pooling: str = "att",
        hidden: int = 256,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
        attn_dim: int = 128,
    ):
        super().__init__()
        if layers not in ("weighted", "last"):
            raise ValueError(f"accent.layers must be 'weighted' or 'last', got {layers!r}")
        if pooling not in ("att", "asp"):
            raise ValueError(f"accent.pooling must be 'att' or 'asp', got {pooling!r}")
        self.layers, self.pooling, self.num_layers = layers, pooling, num_layers
        self.label_smoothing = float(label_smoothing)
        if layers == "weighted":
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))  # softmax(0,...,0) = uniform
        self.att = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        pooled = dim * (2 if pooling == "asp" else 1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(pooled), nn.Linear(pooled, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )
        self.register_buffer("class_weight", torch.ones(num_classes))

    # ----------------------------------------------------------------------------- setup
    def set_class_weights(self, counts: Sequence[int]) -> None:
        """Inverse-frequency weights from TRAIN counts, normalised to mean 1 (so the loss scale stays comparable)."""
        c = torch.tensor([max(int(x), 1) for x in counts], dtype=torch.float32)
        w = c.sum() / (len(c) * c)
        self.class_weight.copy_(w / w.mean())

    def layer_weights(self) -> Optional[torch.Tensor]:
        """Softmax mixing weights over the blocks (None for layers='last'); for logging / the report."""
        return torch.softmax(self.layer_logits.detach().float(), 0) if self.layers == "weighted" else None

    # --------------------------------------------------------------------------- forward
    def _mix(self, layer_outs: List[torch.Tensor], last: torch.Tensor) -> torch.Tensor:
        if self.layers == "last":
            return last.float()
        if len(layer_outs) != self.num_layers:
            raise RuntimeError(f"expected {self.num_layers} block outputs, captured {len(layer_outs)} "
                               "(is encoder_layerdrop > 0? the accent head needs every block to run)")
        w = torch.softmax(self.layer_logits.float(), 0)
        # Blocks have residual streams of different scale: normalise each before weighting, otherwise the
        # largest-scale block dominates the sum whatever the learned weights are.
        return sum(w[i] * F.layer_norm(h.float(), h.shape[-1:]) for i, h in enumerate(layer_outs))

    def forward(self, layer_outs: List[torch.Tensor], last: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """layer_outs: 12 x (B,T',D); last: final encoder output (B,T',D); lens: valid frames (B,). -> logits (B,C)."""
        x = self._mix(layer_outs, last)                                        # (B,T',D) fp32
        valid = torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]
        score = self.att(x).squeeze(-1).masked_fill(~valid, float("-inf"))
        a = torch.softmax(score, dim=-1).unsqueeze(-1)                        # (B,T',1), zero on padding
        mean = (a * x).sum(1)
        if self.pooling == "asp":
            std = ((a * (x - mean[:, None, :]) ** 2).sum(1)).clamp_min(1e-5).sqrt()
            mean = torch.cat([mean, std], dim=-1)
        return self.mlp(mean)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), labels, weight=self.class_weight, label_smoothing=self.label_smoothing)
