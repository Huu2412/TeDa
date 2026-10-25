#!/usr/bin/env python
"""Evaluate a trained checkpoint: WER/CER overall, per region, per text group, bootstrap CI.

    python scripts/evaluate.py --exp e00_baseline --split valid
    python scripts/evaluate.py --exp e00_baseline --split valid --chunk 64,128,128    # chunked decoding
    python scripts/evaluate.py --exp e03_full --split test --final                    # test: final numbers only

The test split is refused unless --final is given: choose K, loss weights, etc. on VALID only.
Outputs next to the checkpoint: results_<split>.json, hyps_<split>.tsv, errors_<split>.json
(the last one feeds paired-bootstrap comparisons between experiments).
"""

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from teda.data.tokenizer import BPETokenizer
from teda.data.vimd_dataset import VimdDataset
from teda.engine.evaluator import evaluate
from teda.models.teda_model import build_model
from teda.utils import dist as D
from teda.utils.ckpt import load_ckpt, write_json
from teda.utils.config import load_config
from teda.utils.logging import setup_logging
from teda.utils.manifest import write_tsv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out-dir", default=None, help="run folder (default <paths.out_root>/<exp_name>)")
    ap.add_argument("--checkpoint", default=None, help="default <out-dir>/best.pt")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--chunk", default=None, help="chunked decoding 'chunk,left,right' (e.g. 64,128,128); default full context")
    ap.add_argument("--batch-frames", type=int, default=None)
    a = ap.parse_args()

    if a.split == "test" and not a.final:
        raise SystemExit("Refusing to evaluate on the test split without --final (test is for the final report only).")

    cfg = load_config(a.config, a.exp, a.set)
    cfg.encoder.pretrained = False  # weights come from the checkpoint; the bundle is only needed for config + CMVN
    ctx = D.setup_distributed()
    run = Path(a.out_dir or Path(cfg.paths.out_root) / cfg.exp_name)
    log = setup_logging(ctx.rank)
    ckpt_path = Path(a.checkpoint or run / "best.pt")
    chunk = [int(x) for x in a.chunk.split(",")] if a.chunk else cfg.get("eval", {}).get("chunk")

    tok = BPETokenizer(cfg.data.tokenizer_model)
    ds = VimdDataset(cfg.data.data_dir, cfg.data.manifest, a.split, tok, cfg.features, train=False,
                     text_field=cfg.data.text_field, limit=cfg.data.get("limit"))
    model, _ = build_model(cfg, tok.vocab_size)
    ck = load_ckpt(ckpt_path)
    model.load_state_dict(ck["model"])
    model.to(ctx.device)
    log.info("checkpoint %s (step %s, valid WER %s); split=%s; chunk=%s", ckpt_path, ck.get("step"), ck.get("valid_wer"), a.split, chunk)

    summary, recs = evaluate(model, ds, tok, ctx, a.batch_frames or int(cfg.eval.batch_frames),
                             amp=ctx.device.type == "cuda", chunk=chunk, num_workers=int(cfg.train.get("num_workers", 2)))
    if ctx.is_main:
        tag = a.split + (f"_chunk{'-'.join(map(str, chunk))}" if chunk else "")
        errors = summary.pop("_errors")
        summary.update({"checkpoint": str(ckpt_path), "ckpt_step": ck.get("step"), "split": a.split, "chunk": chunk, "final": a.final})
        write_json(run / f"results_{tag}.json", summary)
        write_json(run / f"errors_{tag}.json", errors)
        write_tsv(run / f"hyps_{tag}.tsv", ({**r, "groups": ",".join(r["groups"])} for r in recs),
                  ["utt_id", "region_id", "groups", "ref", "hyp"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    D.cleanup()


if __name__ == "__main__":
    main()
