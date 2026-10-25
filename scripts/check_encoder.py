#!/usr/bin/env python
"""FIRST thing to run on Kaggle: verify the real ViP-VL bundle loads into the vendored encoder.

    python scripts/check_encoder.py                 # downloads khanhld/vip-vl-base-vie
    python scripts/check_encoder.py --bundle /path/to/folder

Prints the bundle config, how many checkpoint tensors matched (key prefix handling), parameter counts,
and runs a forward pass in train (dynamic chunk) and eval (full / chunked) mode. It fails loudly if the
checkpoint does not fit the architecture, instead of letting training start from random weights.
"""

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

import teda  # noqa: F401
from teda.data.features import subsampled_length
from teda.models.encoder import SpeechEncoder, build_chunkformer, load_pretrained, read_bundle_config, resolve_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="khanhld/vip-vl-base-vie")
    ap.add_argument("--fallback-config", default="configs/vipvl_encoder.yaml")
    a = ap.parse_args()

    bdir = resolve_bundle(a.bundle)
    print("bundle folder:", bdir)
    print("files:", sorted(p.name for p in bdir.iterdir()))
    conf, input_dim = read_bundle_config(bdir, a.fallback_config)
    print("input_dim:", input_dim)
    print("encoder_conf:", json.dumps(conf, indent=1))

    cmvn = bdir / "global_cmvn"
    print("global_cmvn:", "found" if cmvn.exists() else "MISSING")
    net = build_chunkformer(conf, input_dim, cmvn if cmvn.exists() else None)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"encoder parameters: {n_params:,}  ({len(net.encoders)} blocks)")

    rep = load_pretrained(net, bdir / "pytorch_model.pt")
    print(f"loaded {rep['n_loaded']}/{rep['n_target']} tensors (coverage {100 * rep['coverage']:.1f}%), stripped prefix {rep['prefix_stripped']!r}")
    print("  missing    :", rep["missing"][:10])
    print("  unexpected :", rep["unexpected"][:10])
    print("  mismatched :", rep["shape_mismatch"][:10])

    enc = SpeechEncoder(net)
    x = torch.randn(2, 1000, input_dim)
    lens = torch.tensor([1000, 900])
    enc.train()
    out, ol = enc(x, lens)
    print("train-mode output:", tuple(out.shape), ol.tolist(), "expected lengths", [subsampled_length(int(l)) for l in lens])
    enc.eval()
    with torch.no_grad():
        out, ol = enc(x, lens)
        out_c, ol_c = enc(x, lens, chunk=(64, 128, 128))
    print("eval full-context :", tuple(out.shape), ol.tolist())
    print("eval chunked      :", tuple(out_c.shape), ol_c.tolist())
    assert torch.isfinite(out).all() and torch.isfinite(out_c).all(), "non-finite encoder output"
    print("OK")


if __name__ == "__main__":
    main()
