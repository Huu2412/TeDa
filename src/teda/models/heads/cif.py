"""CIF aligner: turn encoder frames into ONE vector per word and compare it with the word's PhoBERT vector.

Training-time only (E02/E03). At inference this module, the PhoBERT cache and the projection are all unused.

    h_t (B,T',D) -> alpha_t = sigmoid(Conv->Linear)          how much of word is "heard" in frame t
                 -> alpha scaled so that sum_t alpha_t = N     (N = number of words of the reference text)
                 -> integrate-and-fire: word j owns the stretch [j, j+1) of the running sum of alpha
                 -> e_j = weighted mean of the frames that overlap that stretch
                 -> z_j = P_audio(e_j)   compared with the frozen PhoBERT vector of word j (cosine)

Everything is dense tensor algebra (no loop over time), exact and differentiable.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def cif_integrate(h: torch.Tensor, alpha: torch.Tensor, n_words: int):
    """Integrate frames into word vectors.

    h:      (B, T, D) frame vectors
    alpha:  (B, T)    non-negative weights, zero on padding; per utterance sum(alpha) should equal its word count
    Returns e (B, N, D) and mass (B, N): the total weight word j collected (1 for every word when sum(alpha) = N).
    Frame t contributes to word j the overlap of [c_{t-1}, c_t] and [j, j+1], c = cumsum(alpha).
    """
    a = alpha.float()
    c = a.cumsum(1)
    c_prev = c - a
    j = torch.arange(n_words, device=h.device, dtype=a.dtype)
    lo = torch.maximum(c_prev.unsqueeze(-1), j)
    hi = torch.minimum(c.unsqueeze(-1), j + 1.0)
    w = (hi - lo).clamp_min(0.0)                                   # (B, T, N)
    mass = w.sum(1)                                                # (B, N)
    e = torch.einsum("btn,btd->bnd", w, h.float()) / mass.clamp_min(1e-6).unsqueeze(-1)
    return e, mass


class CIFAligner(nn.Module):
    def __init__(self, dim: int, text_dim: int = 768, kernel: int = 3, proj_hidden: int = 1024, dropout: float = 0.1,
                 target_mode: str = "phobert"):
        super().__init__()
        if target_mode not in ("phobert", "random_word", "shuffle"):
            raise ValueError(f"model.align.target_mode must be phobert | random_word | shuffle, got {target_mode!r}")
        self.target_mode = target_mode
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.alpha_out = nn.Linear(dim, 1)
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(proj_hidden, text_dim))
        nn.init.zeros_(self.alpha_out.weight)
        nn.init.zeros_(self.alpha_out.bias)   # alpha starts at 0.5 everywhere; the count scaling fixes the total anyway

    def alphas(self, h: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        mask = (torch.arange(h.size(1), device=h.device)[None] < lens[:, None]).to(h.dtype)
        x = self.conv((h * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        a = torch.sigmoid(self.alpha_out(F.gelu(x)).squeeze(-1).float())
        return a * mask.float()

    def _targets(self, targets: torch.Tensor, wmask: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        if self.target_mode == "shuffle" and wmask.any():
            # control: every word is paired with the PhoBERT vector of a RANDOM word of the batch (breaks the semantics,
            # keeps the vector statistics). If this trains as well as the real targets, PhoBERT contributes nothing.
            idx = wmask.nonzero(as_tuple=True)
            rows = t[idx]
            t = t.clone()
            t[idx] = rows[torch.randperm(rows.size(0), device=rows.device)]
        return t

    def forward(self, h: torch.Tensor, lens: torch.Tensor, targets: torch.Tensor, n_words: torch.Tensor) -> Dict:
        """Returns loss_align, loss_qty (scalars with graph) and detached diagnostics."""
        B, T, _ = h.shape
        n = n_words.to(h.device)
        alpha = self.alphas(h, lens)
        sum_a = alpha.sum(1)
        # word count must fit: one frame per word at least; otherwise this utterance is left out of the loss
        valid = (n > 0) & (n <= lens)
        n_max = max(int(n.max()), 1)
        n_max = min(n_max, targets.size(1))
        alpha_s = alpha * (n.float() / sum_a.clamp_min(1e-4)).unsqueeze(1)      # sum_t alpha_s = N
        e, mass = cif_integrate(h, alpha_s, n_max)
        z = self.proj(e.to(h.dtype) if h.dtype != torch.float32 else e).float()
        wmask = (torch.arange(n_max, device=h.device)[None] < n.clamp_max(n_max)[:, None]) & valid[:, None]
        tgt = self._targets(targets[:, :n_max], wmask)

        cos = F.cosine_similarity(z, tgt, dim=-1)                              # (B, N)
        cnt = wmask.sum().clamp_min(1)
        loss_align = ((1.0 - cos) * wmask).sum() / cnt
        qty = (sum_a - n.float()).abs() / n.float().clamp_min(1.0)
        loss_qty = qty[valid].mean() if valid.any() else sum_a.sum() * 0.0
        if not valid.any():
            loss_align = z.sum() * 0.0

        with torch.no_grad():
            zn, tn = F.normalize(z, dim=-1), F.normalize(tgt, dim=-1)
            sim = torch.bmm(zn, tn.transpose(1, 2))                            # (B, N, N): audio word i vs text word j
            sim = sim.masked_fill(~wmask.unsqueeze(1), float("-inf"))
            hit = (sim.argmax(-1) == torch.arange(n_max, device=h.device)[None]) & wmask
            diag = {
                "align_cos": (cos.detach() * wmask).sum() / cnt,
                "align_top1": hit.sum().float() / cnt,
                "qty_err": qty[valid].mean() if valid.any() else qty.sum() * 0.0,
                "sum_alpha": sum_a[valid].mean() if valid.any() else sum_a.sum() * 0.0,
                "align_words": wmask.sum().float(),
                "align_utts": valid.sum().float(),
                "align_cos_sum": (cos.detach() * wmask).sum(),
                "align_hit_sum": hit.sum().float(),
                "qty_sum": qty[valid].sum() if valid.any() else qty.sum() * 0.0,
            }
        return {"loss_align": loss_align, "loss_qty": loss_qty, **diag}
