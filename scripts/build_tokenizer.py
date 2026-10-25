#!/usr/bin/env python
"""Step 1a: train the BPE-1024 tokenizer on the TRAIN split's normalised text (never valid/test).

    python scripts/build_tokenizer.py --data-dir /kaggle/working/vimd16k

Writes artifacts/tokenizer/bpe1024.model (+ .vocab): small, so commit it to the repo.
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from teda.data.tokenizer import BPETokenizer, train_bpe
from teda.utils.config import load_config
from teda.utils.manifest import read_tsv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data-dir", default=None, help="folder with manifest.tsv (default: data.data_dir in the config)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.vocab_size=512")
    a = ap.parse_args()

    cfg = load_config(a.config, overrides=a.set)
    data_dir = Path(a.data_dir or cfg.data.data_dir)
    rows = read_tsv(data_dir / cfg.data.manifest)
    field = cfg.data.text_field
    train_text = [r[field] for r in rows if r["split"] == "train"]
    print(f"{len(train_text)} train transcripts ({field})")

    model = train_bpe(train_text, Path(cfg.data.tokenizer_model).with_suffix(""), cfg.data.vocab_size)
    tok = BPETokenizer(model)
    print(f"wrote {model}  vocab={tok.vocab_size}")
    for split in ("train", "valid", "test"):
        texts = [r[field] for r in rows if r["split"] == split]
        if texts:
            print(f"  {split:<5} unk rate = {100 * tok.unk_rate(texts):.4f}%  ({len(texts)} utts)")
    sample = train_text[0]
    ids = tok.encode(sample)
    print("example:", sample[:80], "->", ids[:12], "... ->", tok.decode(ids)[:80])
    assert tok.decode(ids) == " ".join(sample.split()), "tokenizer round-trip failed"


if __name__ == "__main__":
    main()
