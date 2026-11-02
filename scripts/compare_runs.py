#!/usr/bin/env python
"""Compare two finished/interrupted runs from their metrics.jsonl (no GPU needed).

    python scripts/compare_runs.py outputs/e00_full_ft outputs/e01_accent

Prints: the settings that differ between the two runs (so you do not compare unequal runs), valid WER (overall and per region) side by side at each evaluation, the best point of each run,
and, for runs with an accent head, accent accuracy / macro-F1, the learned block weights and the gradient
probe summary (accent share of the gradient next to CTC's, and their cosine).
"""

import argparse
import json
from pathlib import Path
from statistics import mean

import yaml


def load(run: str):
    rows = [json.loads(line) for line in open(Path(run) / "metrics.jsonl", encoding="utf-8") if line.strip()]
    return {k: [r for r in rows if r["kind"] == k] for k in ("train", "valid", "grad_probe")}


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(a: str, b: str) -> None:
    """Which settings differ between the two runs? A fair comparison changes ONE thing."""
    try:
        ca, cb = (flat(yaml.safe_load(open(Path(r) / "config.yaml", encoding="utf-8"))) for r in (a, b))
    except OSError:
        print("(config.yaml not found in one of the run folders: cannot compare settings)\n")
        return
    skip = ("exp_name", "paths.", "data.data_dir")
    diff = [(k, ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k) and not k.startswith(skip)]
    print("Settings that differ between A and B:")
    for k, x, y in diff:
        print(f"  {k:<34} A={x!r:<22} B={y!r}")
    if not diff:
        print("  (none)")
    keys = ("train.max_frames_per_batch", "train.accum_grad", "train.epochs", "train.lr_encoder", "train.warmup_steps")
    if any(ca.get(k) != cb.get(k) for k in keys):
        print("  -> WARNING: batch size / schedule differs. The runs do not get the same number of optimizer steps,")
        print("     so a WER gap may come from training budget rather than from the feature under test.")
    print()


def fmt(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="baseline run dir, e.g. outputs/e00_full_ft")
    ap.add_argument("b", help="run to compare, e.g. outputs/e01_accent")
    args = ap.parse_args()
    config_diff(args.a, args.b)
    A, B = load(args.a), load(args.b)
    va, vb = {r["epoch"]: r for r in A["valid"]}, {r["epoch"]: r for r in B["valid"]}
    names = ["North", "Central", "South"]

    print(f"{'epoch':>5} | {'A step':>6} {'A WER%':>7} | {'B step':>6} {'B WER%':>7} {'B-A':>6} | "
          f"{'A Cen%':>7} {'B Cen%':>7} | {'B acc%':>6} {'B F1%':>6}")
    for ep in sorted(set(va) | set(vb)):
        a, b = va.get(ep), vb.get(ep)
        d = None if not (a and b) else b["wer"] - a["wer"]
        cen = lambda r: None if not r else r["by_region"].get("Central")
        print(f"{ep:>5} | {a['step'] if a else '-':>6} {fmt(a and a['wer']):>7} | {b['step'] if b else '-':>6} {fmt(b and b['wer']):>7} "
              f"{('%+.2f' % (100 * d)) if d is not None else '-':>6} | {fmt(cen(a)):>7} {fmt(cen(b)):>7} | "
              f"{fmt(b and b.get('accent_acc'), 1):>6} {fmt(b and b.get('accent_macro_f1'), 1):>6}")

    for tag, run in (("A", A), ("B", B)):
        if not run["valid"]:
            continue
        best = min(run["valid"], key=lambda r: r["wer"])
        last = run["valid"][-1]
        reg = " ".join(f"{n}={fmt(best['by_region'].get(n))}" for n in names)
        print(f"\n{tag}: {len(run['valid'])} evaluations, last step {last['step']} (epoch {last['epoch']}) | "
              f"best WER {fmt(best['wer'])}% at step {best['step']} (epoch {best['epoch']}) | {reg} | last WER {fmt(last['wer'])}%")
        tr = run["train"]
        if tr:
            print(f"   train ctc: first {tr[0]['loss_ctc']:.1f} -> last {tr[-1]['loss_ctc']:.1f}; "
                  f"max grad_norm {max(r['grad_norm'] for r in tr):.1f}")

    if B["valid"] and "accent_layer_weights" in B["valid"][-1]:
        w = B["valid"][-1]["accent_layer_weights"]
        print("\nB accent block weights (last eval, blocks 0..%d): %s" % (len(w) - 1, " ".join(f"{x:.3f}" for x in w)))
        print("   (all ~%.3f = the weights barely moved; a peak shows which block carries accent information)" % (1 / len(w)))
    if B["grad_probe"]:
        gp = B["grad_probe"]
        print(f"\nB gradient probe ({len(gp)} samples): accent share of gradient mean {100 * mean(r['acc_share'] for r in gp):.1f}% "
              f"(min {100 * min(r['acc_share'] for r in gp):.1f}%, max {100 * max(r['acc_share'] for r in gp):.1f}%), "
              f"cos(g_ctc, g_acc) mean {mean(r['cos'] for r in gp):+.2f}")
        print("   cos < 0: accent and CTC pull the encoder in opposite directions; share ~0: accent has no effect on the encoder")


if __name__ == "__main__":
    main()
