#!/usr/bin/env python
"""Step 0b: normalise transcripts, write manifest.tsv and review reports.

    python scripts/scan_text.py --manifest-raw /kaggle/working/vimd16k/manifest_raw.tsv \
        --out-dir outputs/scan --manifest-out /kaggle/working/vimd16k/manifest.tsv

Outputs (in --out-dir):
    norm_report.tsv     every conversion (number/date/time/unit/...) with count + example
    unresolved.tsv      acronyms, foreign words, letter+digit tokens, dropped symbols
    review_sample.tsv   random utterances whose digits were converted (spot-check these)
    stats.json          sizes, hours, region balance, digit/acronym rates, tone + number-word variants
and --manifest-out gets the final manifest (adds region_id, text_clean, text_norm, ...).
Nothing is ever dropped: every utterance of the raw manifest is kept.
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from teda.data.label_maps import REGIONS, region_to_id
from teda.data.text_norm import Normalizer, tone_style_counts
from teda.utils.manifest import FULL_FIELDS, read_tsv, write_tsv

CONVERSION_KINDS = {
    "int", "int_grouped", "int_grouped_en", "decimal", "decimal_dot", "decimal_3digit", "digitwise",
    "digitwise_long", "date", "time", "time_range", "range", "unit", "weekday", "month", "abbrev", "fallback",
}
UNRESOLVED_KINDS = {"acronym", "foreign", "alnum", "drop", "residual_digit", "big_int"}
FLAG_KINDS = {"decimal_dot", "decimal_3digit", "digitwise_long", "int_grouped_en", "fallback"}  # ambiguous -> review

# How the corpus itself writes numbers in letters (helps choose configs/text_norm.yaml).
VARIANTS = {
    "linh vs lẻ (105)": (r"\blinh\b", r"\blẻ\b"),
    "nghìn vs ngàn": (r"\bnghìn\b", r"\bngàn\b"),
    "mươi tư vs mươi bốn": (r"mươi tư\b", r"mươi bốn\b"),
    "mươi mốt vs mươi một": (r"mươi mốt\b", r"mươi một\b"),
    "mươi lăm vs mươi năm": (r"mươi lăm\b", r"mươi năm\b"),
    "tỷ vs tỉ": (r"\btỷ\b", r"\btỉ\b"),
}


def pct(a, b):
    return round(100.0 * a / b, 2) if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-raw", required=True)
    ap.add_argument("--out-dir", default="outputs/scan")
    ap.add_argument("--manifest-out", default=None, help="default: <manifest-raw dir>/manifest.tsv")
    ap.add_argument("--config", default="configs/text_norm.yaml")
    ap.add_argument("--review-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = Path(a.manifest_out) if a.manifest_out else Path(a.manifest_raw).with_name("manifest.tsv")

    norm = Normalizer.from_config(a.config)
    rows = read_tsv(a.manifest_raw)
    print(f"{len(rows)} utterances; tone_style={norm.tone_style}; abbreviations={len(norm.abbreviations)}")

    events = defaultdict(lambda: {"count": 0, "utts": set(), "example": ""})  # (kind, src, dst)
    utts_with_kind = defaultdict(set)
    per_split = defaultdict(lambda: Counter())
    hours = Counter()
    regions = defaultdict(Counter)
    words_len = []
    variant_counts = {k: [0, 0] for k in VARIANTS}
    tone_old = tone_new = 0
    unknown_regions = Counter()
    out_rows, review_pool = [], []

    for r in rows:
        raw = r["text_raw"]
        trace = []
        text_norm = norm.normalize(raw, trace)
        text_clean = norm.clean_for_phobert(raw)
        has_digit = any(c.isdigit() for c in raw)
        try:
            region_id = region_to_id(r["region"])
        except ValueError:
            unknown_regions[r["region"]] += 1
            region_id = -1

        sp = r["split"]
        per_split[sp]["utts"] += 1
        hours[sp] += float(r["duration"]) / 3600
        regions[sp][r["region"]] += 1
        if has_digit:
            per_split[sp]["with_digit"] += 1
        for kind, src, dst in trace:
            e = events[(kind, src, dst)]
            e["count"] += 1
            e["utts"].add(r["utt_id"])
            e["example"] = e["example"] or r["utt_id"]
            utts_with_kind[(sp, kind)].add(r["utt_id"])
        words_len.append(len(text_norm.split()))

        low = unicodedata.normalize("NFC", raw).lower()
        for k, (p1, p2) in VARIANTS.items():
            variant_counts[k][0] += len(re.findall(p1, low))
            variant_counts[k][1] += len(re.findall(p2, low))
        o, n = tone_style_counts(raw)
        tone_old, tone_new = tone_old + o, tone_new + n

        changed = text_norm != " ".join(raw.lower().split())
        out_rows.append({**r, "region_id": region_id, "text_clean": text_clean, "text_norm": text_norm,
                         "has_digit": int(has_digit), "norm_changed": int(changed)})
        if has_digit and any(k in CONVERSION_KINDS for k, _, _ in trace):
            review_pool.append({"utt_id": r["utt_id"], "split": sp, "text_raw": raw, "text_norm": text_norm})

    if unknown_regions:
        raise SystemExit(
            f"Unknown region values {dict(unknown_regions)}. Update teda/data/label_maps.py "
            "(_ALIASES) with the right spelling and re-run."
        )

    # ------------------------------------------------------------------ reports
    def table(kinds):
        recs = [
            {"kind": k, "count": e["count"], "n_utts": len(e["utts"]), "flag": "REVIEW" if k in FLAG_KINDS else "",
             "source": s, "spoken": d, "example_utt": e["example"]}
            for (k, s, d), e in events.items() if k in kinds
        ]
        recs.sort(key=lambda x: (-x["count"], x["kind"], x["source"]))
        return recs

    write_tsv(out_dir / "norm_report.tsv", table(CONVERSION_KINDS),
              ["kind", "flag", "source", "spoken", "count", "n_utts", "example_utt"])
    write_tsv(out_dir / "unresolved.tsv", table(UNRESOLVED_KINDS),
              ["kind", "source", "count", "n_utts", "example_utt"])
    random.Random(a.seed).shuffle(review_pool)
    write_tsv(out_dir / "review_sample.tsv", review_pool[: a.review_n], ["utt_id", "split", "text_raw", "text_norm"])

    words_sorted = sorted(words_len)
    n = len(words_sorted)
    stats = {
        "n_utts": n,
        "tone_style_in_corpus": {"old(hòa)": tone_old, "new(hoà)": tone_new, "configured": norm.tone_style},
        "number_word_variants_in_corpus": {k: {"first": v[0], "second": v[1]} for k, v in variant_counts.items()},
        "words_per_utt_norm": {"mean": round(sum(words_sorted) / max(n, 1), 1),
                               "p95": words_sorted[int(0.95 * (n - 1))] if n else 0,
                               "max": words_sorted[-1] if n else 0,
                               "over_150": sum(w > 150 for w in words_sorted)},
        "splits": {},
        "warnings": [],
    }
    for sp, c in per_split.items():
        kinds_n = lambda ks: len(set().union(*[utts_with_kind[(sp, k)] for k in ks]))  # noqa: E731
        stats["splits"][sp] = {
            "utts": c["utts"], "hours": round(hours[sp], 2),
            "with_digit": c["with_digit"], "with_digit_pct": pct(c["with_digit"], c["utts"]),
            "with_acronym": kinds_n(["acronym"]), "with_foreign_letters": kinds_n(["foreign"]),
            "with_alnum_token": kinds_n(["alnum"]),
            "regions": dict(regions[sp]),
        }
        missing = [x for x in REGIONS if x not in {k for k in regions[sp]}]
        if missing:
            stats["warnings"].append(f"split '{sp}' has no utterance for regions {missing} (check region spelling)")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    write_tsv(manifest_out, out_rows, FULL_FIELDS)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote: {manifest_out}\n       {out_dir}/norm_report.tsv, unresolved.tsv, review_sample.tsv, stats.json")
    print("Next: read unresolved.tsv (fill configs/abbreviations.tsv), spot-check review_sample.tsv, "
          "choose number style from stats.json, then re-run this script.")


if __name__ == "__main__":
    main()
