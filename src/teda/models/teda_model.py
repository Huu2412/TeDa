"""TedaModel: speech encoder + CTC head (step 1 / E00). Accent and alignment heads plug in here later."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from teda.models.encoder import (
    SpeechEncoder, build_chunkformer, load_pretrained, read_bundle_config, resolve_bundle,
)
from teda.models.heads.accent_head import AccentHead
from teda.models.heads.ctc_head import CTCHead
from teda.utils.freeze import apply_freeze, format_freeze_report, freeze_report

log = logging.getLogger("teda")


class TedaModel(nn.Module):
    def __init__(
        self,
        encoder: SpeechEncoder,
        ctc: CTCHead,
        frozen_modules: Sequence[nn.Module] = (),
        accent: Optional[AccentHead] = None,
        weight_accent: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.accent = accent
        self.weight_accent = float(weight_accent)
        self._frozen = list(frozen_modules)  # plain list on purpose: these are already registered via encoder

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._frozen:  # frozen blocks never use dropout; the encoder itself stays in train mode
            m.eval()
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc, lens = self.encoder(batch["feats"], batch["feat_lens"])
        loss_ctc = self.ctc.loss(enc, lens, batch["tokens"], batch["token_lens"])
        out = {"loss": loss_ctc, "loss_ctc": loss_ctc.detach()}
        if self.accent is not None and self.weight_accent > 0 and "region_ids" in batch:
            loss_acc, acc_acc = self.accent.loss_and_acc(enc, lens, batch["region_ids"])
            out["loss"] = out["loss"] + self.weight_accent * loss_acc
            out["loss_accent"] = loss_acc.detach()
            out["accent_acc"] = acc_acc.detach()
        return out

    @torch.no_grad()
    def encode_logits(self, feats: torch.Tensor, feat_lens: torch.Tensor, chunk: Optional[Sequence[int]] = None):
        enc, lens = self.encoder(feats, feat_lens, chunk=chunk)
        return self.ctc.logits(enc), lens


def build_model(cfg, vocab_size: int, blank_id: int = 0) -> Tuple[TedaModel, Dict]:
    """Build encoder (+pretrained weights) -> freeze -> CTC head. Returns (model, freeze report)."""
    ec = cfg.encoder
    if ec.get("encoder_conf"):  # explicit architecture (unit tests / ablations), no bundle
        conf, input_dim, cmvn_file, bundle_dir = dict(ec.encoder_conf), int(ec.get("input_dim", 80)), None, None
    else:
        bundle_dir = resolve_bundle(ec.bundle)
        conf, input_dim = read_bundle_config(bundle_dir, ec.get("fallback_config"))
        cmvn_file = bundle_dir / "global_cmvn"
        if not cmvn_file.exists():
            raise FileNotFoundError(f"{cmvn_file} missing: the encoder needs the CMVN stats it was pretrained with")
    conf.update(dict(ec.get("overrides") or {}))

    net = build_chunkformer(conf, input_dim, cmvn_file)
    if bundle_dir is not None and ec.get("pretrained", True):
        ckpt = bundle_dir / "pytorch_model.pt"
        rep = load_pretrained(net, ckpt, float(ec.get("min_match", 0.95)))
        log.info("pretrained %s: %d/%d tensors loaded (coverage %.1f%%, stripped prefix %r); %d unexpected, %d shape mismatches",
                 ckpt.name, rep["n_loaded"], rep["n_target"], 100 * rep["coverage"], rep["prefix_stripped"],
                 len(rep["unexpected"]), len(rep["shape_mismatch"]))
        if rep["missing"]:
            log.info("  not in checkpoint (randomly initialised): %s%s", rep["missing"][:6], " ..." if len(rep["missing"]) > 6 else "")
    elif bundle_dir is not None:
        log.warning("encoder.pretrained=false: training the encoder from random initialisation")

    frozen = apply_freeze(net, bool(ec.get("freeze_embed", False)), ec.get("freeze_layers") or [], bool(ec.get("freeze_after_norm", False)))
    encoder = SpeechEncoder(net)
    ctc = CTCHead(encoder.output_size, vocab_size, blank_id, float(cfg.get("model", {}).get("ctc_dropout", 0.0)))

    accent_cfg = cfg.get("model", {}).get("accent") or {}
    accent = None
    weight_accent = float(accent_cfg.get("weight", 0.0))
    if accent_cfg.get("enabled", False) or weight_accent > 0:
        num_classes = int(accent_cfg.get("num_classes", len(cfg.get("data", {}).get("regions", ["North", "Central", "South"]))))
        dropout = float(accent_cfg.get("dropout", 0.1))
        accent = AccentHead(encoder.output_size, num_classes=num_classes, dropout=dropout)
        if weight_accent <= 0:
            weight_accent = 0.2  # default weight
        log.info("accent head enabled: num_classes=%d, weight=%.2f, dropout=%.2f", num_classes, weight_accent, dropout)

    model = TedaModel(encoder, ctc, frozen, accent=accent, weight_accent=weight_accent)
    rep = freeze_report(model, net)
    log.info("parameters (trainable / frozen):\n%s", format_freeze_report(rep))
    return model, rep
