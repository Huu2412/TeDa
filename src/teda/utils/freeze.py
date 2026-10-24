"""Partial freezing of the ViP-VL encoder.

ChunkFormerEncoder layout (12 blocks in ViP-VL base):
    embed          depthwise-conv subsampling front-end (acts as the "feature extractor")
    encoders[0..11] conformer blocks
    after_norm     final LayerNorm

Frozen modules get requires_grad=False AND are kept in eval() mode (no dropout), but ONLY those
submodules: the encoder itself stays in train mode so that its random dynamic-chunk selection
(encoder.forward picks chunk/context sizes when self.training) keeps working.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def apply_freeze(
    encoder: torch.nn.Module,
    freeze_embed: bool = False,
    freeze_layers: Sequence[int] = (),
    freeze_after_norm: bool = False,
) -> List[torch.nn.Module]:
    """Freeze the selected parts of a ChunkFormerEncoder. Returns the frozen modules."""
    n_layers = len(encoder.encoders)
    layers = sorted({int(i) for i in (freeze_layers or [])})
    bad = [i for i in layers if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"freeze_layers {bad} out of range for an encoder with {n_layers} layers")

    targets: List[torch.nn.Module] = []
    if freeze_embed:
        targets.append(encoder.embed)
    targets += [encoder.encoders[i] for i in layers]
    if freeze_after_norm:
        targets.append(encoder.after_norm)

    for m in targets:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    return targets


def count_params(module: torch.nn.Module) -> Tuple[int, int]:
    """(trainable, frozen) parameter counts."""
    tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return tr, fr


def freeze_report(model: torch.nn.Module, encoder: torch.nn.Module) -> Dict[str, Tuple[int, int]]:
    """Per-component (trainable, frozen) counts, for the log and for the report table."""
    rep: Dict[str, Tuple[int, int]] = {"embed": count_params(encoder.embed)}
    for i, layer in enumerate(encoder.encoders):
        rep[f"layer{i:02d}"] = count_params(layer)
    rep["after_norm"] = count_params(encoder.after_norm)
    enc_total = (sum(v[0] for v in rep.values()), sum(v[1] for v in rep.values()))
    rep["encoder_total"] = enc_total
    tr_all, fr_all = count_params(model)
    rep["head_and_other"] = (tr_all - enc_total[0], fr_all - enc_total[1])
    rep["model_total"] = (tr_all, fr_all)
    return rep


def format_freeze_report(rep: Dict[str, Tuple[int, int]]) -> str:
    lines = [f"{'component':<16}{'trainable':>12}{'frozen':>12}"]
    for name, (tr, fr) in rep.items():
        if name.startswith("layer") and tr + fr == 0:
            continue
        lines.append(f"{name:<16}{tr:>12,}{fr:>12,}")
    return "\n".join(lines)
