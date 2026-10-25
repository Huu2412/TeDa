#!/usr/bin/env python
"""Train one experiment.

    MALLOC_ARENA_MAX=2 torchrun --nproc_per_node=2 scripts/train.py --exp e00_baseline
    python scripts/train.py --exp e00_baseline --set train.epochs=1 data.limit=64     # 1-process smoke test

Re-running the same command resumes from outputs/<exp>/last.pt; a finished run (done.json) is a no-op.

MEMORY NOTES
------------
MALLOC_ARENA_MAX=2: glibc by default creates up to 8*nCPU malloc arenas (memory pools). Each arena
retains its own free-list and does not return pages to the OS until the arena is destroyed. On a
Kaggle 2-CPU node the default can create ~16 arenas, bloating RSS by hundreds of MB. Setting this
to 2 before any C library is loaded limits fragmentation without affecting performance.

cuDNN plan cache: PyTorch / cuDNN keeps one convolution plan per unique tensor shape in host RAM.
With dynamic audio batching, thousands of distinct (B, T, C) shapes may appear over a training run,
especially after unfreezing all conformer blocks. This script caps the cache size at startup.
Setting train.pad_multiple=128 in the config is the complementary fix: it collapses the number of
distinct shapes to a few dozen.
"""

import os
# Set MALLOC_ARENA_MAX before importing any C-backed library so that glibc uses at most 2 arenas.
# This must happen as early as possible in the process; once glibc has already allocated arenas
# the env-var change has no effect.
if "MALLOC_ARENA_MAX" not in os.environ:
    os.environ["MALLOC_ARENA_MAX"] = "2"

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from teda.data.tokenizer import BPETokenizer
from teda.data.vimd_dataset import VimdDataset
from teda.engine.trainer import Trainer
from teda.models.encoder import resolve_bundle
from teda.models.teda_model import build_model
from teda.utils import dist as D
from teda.utils.config import dump_config, load_config
from teda.utils.logging import setup_logging
from teda.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="name in configs/exp/ (e.g. e00_baseline) or a yaml path")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 encoder.freeze_layers=[0,1]")
    ap.add_argument("--out-dir", default=None, help="default: <paths.out_root>/<exp_name>")
    a = ap.parse_args()

    cfg = load_config(a.config, a.exp, a.set)
    ctx = D.setup_distributed()
    out = Path(a.out_dir or Path(cfg.paths.out_root) / cfg.exp_name)
    log = setup_logging(ctx.rank, out / "train.log")
    set_seed(cfg.train.seed, ctx.rank)

    if ctx.is_main:
        out.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, out / "config.yaml")
        if not cfg.encoder.get("encoder_conf"):
            resolve_bundle(cfg.encoder.bundle)  # download once; other ranks then hit the cache
    D.barrier()

    tok = BPETokenizer(cfg.data.tokenizer_model)
    common = dict(data_dir=cfg.data.data_dir, manifest=cfg.data.manifest, tokenizer=tok, feat_cfg=cfg.features,
                  text_field=cfg.data.text_field, limit=cfg.data.get("limit"))
    train_ds = VimdDataset(split="train", train=True, max_dur=cfg.train.get("max_dur"),
                           drop_infeasible_ctc=bool(cfg.train.get("drop_infeasible_ctc", True)), **common)
    valid_ds = VimdDataset(split="valid", train=False, **common)
    model, _ = build_model(cfg, tok.vocab_size)
    log.info("world_size=%d device=%s exp=%s out=%s", ctx.world_size, ctx.device, cfg.exp_name, out)

    Trainer(cfg, model, train_ds, valid_ds, tok, ctx, out).fit()
    D.cleanup()


if __name__ == "__main__":
    main()
