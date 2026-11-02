"""TedaModel: speech encoder + CTC head (E00) + optional accent head (E01) + optional CIF/PhoBERT aligner (E02/E03)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from teda.data.label_maps import REGIONS
from teda.models.encoder import (
    SpeechEncoder, build_chunkformer, load_pretrained, read_bundle_config, resolve_bundle,
)
from teda.models.heads.accent_head import AccentHead
from teda.models.heads.cif import CIFAligner
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
        lambda_acc: float = 0.0,
        weighting: str = "relative",
        align: Optional[CIFAligner] = None,
        lambda_align: float = 0.0,
        lambda_qty: float = 0.0,
        align_weighting: str = "relative",
        align_layer: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.accent = accent            # training-time auxiliary head; never used to produce the transcript
        self.lambda_acc = float(lambda_acc)
        if weighting not in ("relative", "fixed"):
            raise ValueError(f"model.accent.weighting must be 'relative' or 'fixed', got {weighting!r}")
        # fixed:    loss = ctc + lambda * accent_ce
        # relative: loss = ctc + lambda * ctc.detach() * accent_ce / ln(C)
        #   CTC here is a per-utterance SUM (tens to hundreds) while the accent CE is O(1), so a fixed lambda
        #   leaves the accent gradient negligible next to CTC's. 'relative' makes lambda a SHARE: at chance level
        #   (CE = ln C) the accent term equals lambda x the CTC term, however large CTC is. The CTC factor is
        #   detached, so it only scales the accent gradient and never pushes CTC itself.
        self.weighting = weighting
        self.chance_ce = math.log(max(accent.mlp[-1].out_features, 2)) if accent is not None else 1.0
        self.aux_scale = 1.0            # 0..1 ramp for the auxiliary loss, set by the trainer every step
        self.accent_detached = False    # True during the warm-up phase: the head learns, the encoder is untouched
        # ---- E02/E03: word-level alignment of encoder frames to frozen PhoBERT word vectors (training only)
        self.align = align
        self.lambda_align, self.lambda_qty = float(lambda_align), float(lambda_qty)
        if align_weighting not in ("relative", "fixed"):
            raise ValueError(f"model.align.weighting must be 'relative' or 'fixed', got {align_weighting!r}")
        self.align_weighting = align_weighting      # relative: term = share x CTC (cosine loss is O(1), CTC is tens-hundreds)
        self.align_layer = align_layer              # None = final encoder output, k = output of block k
        self.align_scale = 1.0                      # 0..1 ramp, set by the trainer every step
        self.align_detached = False                 # warm-up: CIF/projection learn on detached features
        self._capture = accent is not None or (align is not None and align_layer is not None)
        if self._capture:
            encoder.enable_layer_capture()
        self._frozen = list(frozen_modules)  # plain list on purpose: these are already registered via encoder

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._frozen:  # frozen blocks never use dropout; the encoder itself stays in train mode
            m.eval()
        return self

    def _align_input(self, enc: torch.Tensor, outs) -> torch.Tensor:
        return enc if self.align_layer is None else outs[self.align_layer]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc, lens = self.encoder(batch["feats"], batch["feat_lens"])
        loss_ctc = self.ctc.loss(enc, lens, batch["tokens"], batch["token_lens"])
        out = {"loss": loss_ctc, "loss_ctc": loss_ctc.detach()}
        outs = self.encoder.pop_layer_outputs() if self._capture else None
        total = loss_ctc
        if self.accent is not None:
            labels = batch["region_ids"]
            a_outs, last = outs, enc
            if self.accent_detached:
                # Warm-up: the head sees DETACHED features. It already learns to classify (its own parameters get
                # the gradient) but sends nothing into the encoder, so CTC can leave its early "all blank" plateau
                # undisturbed. After model.accent.start_step the features are attached and the weight ramps up.
                a_outs, last = [o.detach() for o in outs], enc.detach()
            logits = self.accent(a_outs, last, lens)
            loss_acc = self.accent.loss(logits, labels)
            if self.accent_detached:
                w = 1.0
            else:
                w = self.lambda_acc * self.aux_scale
                if self.weighting == "relative":
                    w = w * loss_ctc.detach() / self.chance_ce
            term = w * loss_acc
            total = total + term
            out["loss_accent"] = loss_acc.detach()
            out["accent_acc"] = (logits.argmax(-1) == labels).float().mean().detach()
            out["loss_accent_term"] = term.detach()
            out["_graph_ctc"], out["_graph_acc"] = loss_ctc, term   # for the optional gradient probe (trainer)
        if self.align is not None and "word_targets" in batch:
            h = self._align_input(enc, outs)
            if self.align_detached:
                h = h.detach()      # warm-up: the CIF module and projection learn, the encoder is left alone
            al = self.align(h, lens, batch["word_targets"], batch["n_words"])
            if self.align_detached:
                w_a = w_q = 1.0
            else:
                w_a, w_q = self.lambda_align * self.align_scale, self.lambda_qty * self.align_scale
                if self.align_weighting == "relative":    # the cosine / count losses are O(1): scale by the CTC level
                    w_a, w_q = w_a * loss_ctc.detach(), w_q * loss_ctc.detach()
            a_term = w_a * al["loss_align"] + w_q * al["loss_qty"]
            total = total + a_term
            out["loss_align"] = al["loss_align"].detach()
            out["loss_qty"] = al["loss_qty"].detach()
            out["loss_align_term"] = a_term.detach()
            for k in ("align_cos", "align_top1", "qty_err", "sum_alpha"):
                out[k] = al[k]
        out["loss"] = total
        return out

    @torch.no_grad()
    def encode_eval(self, feats, feat_lens, chunk=None, word_targets=None, n_words=None):
        """One encoder pass for validation: CTC logits, plus the accent logits and alignment diagnostics when present."""
        enc, lens = self.encoder(feats, feat_lens, chunk=chunk)
        outs = self.encoder.pop_layer_outputs() if self._capture else None
        res = {"logits": self.ctc.logits(enc), "lens": lens}
        if self.accent is not None:
            res["accent_logits"] = self.accent(outs, enc, lens)
        if self.align is not None and word_targets is not None:
            res["align"] = self.align(self._align_input(enc, outs), lens, word_targets, n_words)
        return res

    @torch.no_grad()
    def encode_logits(self, feats: torch.Tensor, feat_lens: torch.Tensor, chunk: Optional[Sequence[int]] = None):
        enc, lens = self.encoder(feats, feat_lens, chunk=chunk)
        self.encoder.pop_layer_outputs()  # release captured activations
        return self.ctc.logits(enc), lens

    @torch.no_grad()
    def encode_logits_accent(self, feats: torch.Tensor, feat_lens: torch.Tensor, chunk: Optional[Sequence[int]] = None):
        """Like encode_logits, plus the accent head's logits (B, 3). Only for evaluating the auxiliary task."""
        enc, lens = self.encoder(feats, feat_lens, chunk=chunk)
        acc = self.accent(self.encoder.pop_layer_outputs(), enc, lens)
        return self.ctc.logits(enc), lens, acc


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

    ac = cfg.get("model", {}).get("accent") or {}
    accent = None
    if ac.get("enabled", False):
        accent = AccentHead(
            encoder.output_size, encoder.num_blocks, num_classes=int(ac.get("num_classes", len(REGIONS))),
            layers=str(ac.get("layers", "weighted")), pooling=str(ac.get("pooling", "att")),
            hidden=int(ac.get("hidden", 256)), dropout=float(ac.get("dropout", 0.1)),
            label_smoothing=float(ac.get("label_smoothing", 0.1)),
        )
        log.info("accent head: layers=%s pooling=%s weighting=%s lambda=%.3f start_step=%s ramp_steps=%s", accent.layers,
                 accent.pooling, ac.get("weighting", "relative"), float(ac.get("lambda", 0.1)), ac.get("start_step", 0),
                 ac.get("ramp_steps", 0))

    al_cfg = cfg.get("model", {}).get("align") or {}
    align = None
    if al_cfg.get("enabled", False):
        layer = al_cfg.get("layer", "last")
        layer = None if str(layer) == "last" else int(layer)
        if layer is not None and not -encoder.num_blocks <= layer < encoder.num_blocks:
            raise ValueError(f"model.align.layer must be 'last' or a block index in [0, {encoder.num_blocks}), got {layer}")
        align = CIFAligner(encoder.output_size, text_dim=int(al_cfg.get("text_dim", 768)), kernel=int(al_cfg.get("kernel", 3)),
                           proj_hidden=int(al_cfg.get("proj_hidden", 1024)), dropout=float(al_cfg.get("dropout", 0.1)),
                           target_mode=str(al_cfg.get("target_mode", "phobert")))
        log.info("align head (CIF): layer=%s target_mode=%s weighting=%s lambda_align=%.3f lambda_qty=%.3f start_step=%s ramp_steps=%s",
                 "last" if layer is None else layer, align.target_mode, al_cfg.get("weighting", "relative"),
                 float(al_cfg.get("lambda_align", 0.1)), float(al_cfg.get("lambda_qty", 0.05)), al_cfg.get("start_step", 0),
                 al_cfg.get("ramp_steps", 0))

    model = TedaModel(encoder, ctc, frozen, accent=accent, lambda_acc=float(ac.get("lambda", 0.1)),
                      weighting=str(ac.get("weighting", "relative")), align=align,
                      lambda_align=float(al_cfg.get("lambda_align", 0.1)), lambda_qty=float(al_cfg.get("lambda_qty", 0.05)),
                      align_weighting=str(al_cfg.get("weighting", "relative")),
                      align_layer=layer if align is not None else None)
    rep = freeze_report(model, net)
    log.info("parameters (trainable / frozen):\n%s", format_freeze_report(rep))
    return model, rep
