"""ViP-VL encoder = ChunkFormerEncoder (vendored in third_party/) + pretrained weights.

Bundle layout published with the model (huggingface.co/khanhld/vip-vl-base-vie):
    pytorch_model.pt   encoder-only weights
    config.yaml        encoder_conf (+ feature config)
    global_cmvn        CMVN statistics (json) used for the input fbank
`encoder.bundle` in the config is either a local folder with these files or a Hugging Face repo id.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

import teda  # noqa: F401  (puts third_party/chunkformer on sys.path)
from chunkformer.modules.cmvn import GlobalCMVN
from chunkformer.modules.encoder import ChunkFormerEncoder
from chunkformer.utils.cmvn import load_cmvn
from teda.utils.config import load_yaml

log = logging.getLogger("teda")


def resolve_bundle(bundle: str | Path) -> Path:
    p = Path(bundle)
    if p.is_dir():
        return p
    if p.is_file():
        return p.parent
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(bundle)))


def read_bundle_config(bundle_dir: Path, fallback: Optional[str | Path] = None) -> Tuple[Dict, int]:
    """(encoder_conf, input_dim) from the bundle's config.yaml (or the fallback file)."""
    path = bundle_dir / "config.yaml"
    if not path.exists():
        if fallback is None or not Path(fallback).exists():
            raise FileNotFoundError(f"{path} not found and no fallback encoder config given")
        log.warning("bundle has no config.yaml; using fallback %s", fallback)
        path = Path(fallback)
    cfg = load_yaml(path)
    if "encoder_conf" not in cfg:
        raise KeyError(f"{path} has no 'encoder_conf' key (found: {sorted(cfg)})")
    fb = cfg.get("fbank_conf") or (cfg.get("dataset_conf") or {}).get("fbank_conf") or {}
    input_dim = int(cfg.get("input_dim") or fb.get("num_mel_bins") or 80)
    return dict(cfg["encoder_conf"]), input_dim


def build_chunkformer(encoder_conf: Dict, input_dim: int = 80, cmvn_file: Optional[str | Path] = None) -> ChunkFormerEncoder:
    cmvn = None
    if cmvn_file:
        mean, istd = load_cmvn(str(cmvn_file), True)
        cmvn = GlobalCMVN(torch.from_numpy(np.asarray(mean)).float(), torch.from_numpy(np.asarray(istd)).float())
    return ChunkFormerEncoder(input_dim, global_cmvn=cmvn, **encoder_conf)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def load_pretrained(encoder: ChunkFormerEncoder, ckpt_file: str | Path, min_match: float = 0.95) -> Dict:
    """Load encoder weights (strict=False) and REPORT what matched. Refuses to continue if coverage is low,
    so a wrong key prefix / architecture never silently turns into training from scratch."""
    raw = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in _extract_state_dict(raw).items() if torch.is_tensor(v)}
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    target = encoder.state_dict()

    best = None
    for prefix in ("", "encoder.", "model.encoder."):
        mapped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        ok = {k: v for k, v in mapped.items() if k in target and target[k].shape == v.shape}
        if best is None or len(ok) > len(best[1]):
            best = (prefix, ok, mapped)
    prefix, ok, mapped = best

    wanted = [k for k in target if not k.startswith("global_cmvn.")]  # CMVN comes from the bundle file
    loaded = [k for k in wanted if k in ok]
    mismatch = [k for k in mapped if k in target and target[k].shape != mapped[k].shape]
    used = {prefix + k for k in ok} | {prefix + k for k in mismatch}
    report = {
        "prefix_stripped": prefix,
        "n_target": len(wanted),
        "n_loaded": len(loaded),
        "coverage": len(loaded) / max(len(wanted), 1),
        "missing": [k for k in wanted if k not in ok],
        # every checkpoint tensor that was NOT loaded into the encoder, including keys outside the
        # chosen prefix (e.g. pretraining-only quantizer / prediction-head tensors)
        "unexpected": [k for k in sd if k not in used],
        "shape_mismatch": mismatch,
    }
    if report["coverage"] < min_match:
        raise RuntimeError(
            f"only {report['n_loaded']}/{report['n_target']} encoder tensors matched checkpoint {ckpt_file} "
            f"(best prefix {prefix!r}). First missing: {report['missing'][:5]}; "
            f"first unexpected: {report['unexpected'][:5]}; shape mismatches: {report['shape_mismatch'][:5]}. "
            "Check that encoder_conf matches the checkpoint."
        )
    encoder.load_state_dict({k: v for k, v in ok.items() if not k.startswith("global_cmvn.")}, strict=False)
    return report


class SpeechEncoder(nn.Module):
    """Uniform interface: (feats (B,T,80), feat_lens (B)) -> (hidden (B,T',D), hidden_lens (B)).

    Training: the underlying encoder draws a random (chunk, left, right) context from its
    dynamic_*_sizes on every call. Evaluation: full context unless `chunk=(c, l, r)` is given.
    """

    def __init__(self, net: ChunkFormerEncoder):
        super().__init__()
        self.net = net
        self.output_size = net.output_size()

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor, chunk: Optional[Sequence[int]] = None):
        if self.training or chunk is None:
            out, mask = self.net(feats, feat_lens)
        else:
            c, l, r = chunk
            out, mask = self.net(feats, feat_lens, chunk_size=c, left_context_size=l, right_context_size=r)
        out_lens = mask.reshape(mask.size(0), -1).sum(-1).long()
        return out, out_lens
