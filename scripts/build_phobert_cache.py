"""Build the PhoBERT word-vector cache used by E02/E03 (run ONCE, e.g. on a Kaggle GPU session; ~10-20 min).

    pip install transformers            # preinstalled on Kaggle
    python scripts/build_phobert_cache.py --data-dir $DATA_DIR --out /kaggle/working/phobert_cache --unit syllable
    python scripts/build_phobert_cache.py ... --unit word    # needs `pip install py_vncorenlp` and a JDK

Then train with:  --set data.align_cache=/kaggle/working/phobert_cache
Training never loads PhoBERT; it only memory-maps the vectors written here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from teda.data.phobert_cache import make_segmenter, unit_vectors, write_cache  # noqa: E402
from teda.utils.manifest import read_tsv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", default="manifest.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-field", default="text_norm", help="must be the CTC reference text (words are aligned 1:1 with it)")
    ap.add_argument("--unit", choices=["word", "syllable"], default="word")
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--vncorenlp-dir", default="artifacts/vncorenlp")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-pieces", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    rows = read_tsv(Path(a.data_dir) / a.manifest)
    if a.limit:
        rows = rows[: a.limit]
    segment = make_segmenter(a.unit, a.vncorenlp_dir)
    units = [segment(r[a.text_field]) for r in rows]
    print(f"{len(rows)} utterances, {sum(map(len, units))} units ({a.unit}); first: {units[0][:12]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()
    if dev.type == "cuda":
        model = model.half()
    t0 = time.time()
    vecs = unit_vectors(units, tok, model, dev, a.max_pieces, a.batch_size)
    meta = write_cache(a.out, [r["utt_id"] for r in rows], units, vecs, [r["split"] == "train" for r in rows],
                       {"unit": a.unit, "model": a.model, "text_field": a.text_field})
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}: {meta}")


if __name__ == "__main__":
    main()
