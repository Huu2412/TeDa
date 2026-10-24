#!/usr/bin/env python
"""Run the freeze search sequentially and collect VALID WER into a table.

    python scripts/sweep.py                      # all runs in configs/sweep/freeze_search.yaml
    python scripts/sweep.py --only embed_low6    # a single run

Each run is `torchrun ... scripts/train.py` with overrides from the sweep file. Finished runs
(done.json) are skipped, interrupted ones resume from their last.pt, so you can simply re-run this
command in the next Kaggle session. Results: <out_dir>/summary.tsv.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from teda.utils.ckpt import read_json
from teda.utils.manifest import write_tsv


def overrides(d: dict):
    return [f"{k}={json.dumps(v)}" for k, v in d.items()]  # json is valid YAML: [0, 1], true, null, 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="configs/sweep/freeze_search.yaml")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--nproc", type=int, default=None, help="GPUs per run (default: all visible)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[],
                    help="extra overrides applied to EVERY run, e.g. data.data_dir=/kaggle/working/vimd16k train.max_frames_per_batch=24000")
    a = ap.parse_args()

    sw = yaml.safe_load(open(a.sweep, encoding="utf-8"))
    out_root = Path(sw["out_dir"])
    nproc = a.nproc
    if nproc is None:
        import torch
        nproc = max(torch.cuda.device_count(), 1)

    rows = []
    for run in sw["runs"]:
        name = run["name"]
        if a.only and name not in a.only:
            continue
        out = out_root / name
        done = read_json(out / "done.json")
        if not done:
            launcher = ["torchrun", f"--nproc_per_node={nproc}"] if nproc > 1 else [sys.executable]
            cmd = launcher + ["scripts/train.py", "--exp", sw["base_exp"], "--config", a.config, "--out-dir", str(out),
                              "--set", *overrides({**sw.get("common_set", {}), **run.get("set", {})}), *a.set]
            print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd)
            done = read_json(out / "done.json")
            if rc != 0 or not done:
                print(f"run '{name}' did not finish (exit {rc}). Re-run this command to resume it.", flush=True)
                break
        rows.append({"name": name, "valid_wer": done.get("best_valid_wer"), "best_step": done.get("best_step"),
                     "epochs": done.get("epochs"), **{k: json.dumps(v) for k, v in run.get("set", {}).items()}})

    all_runs = [r for r in sw["runs"] if not a.only or r["name"] in a.only]
    cols = ["name", "valid_wer", "best_step", "epochs"] + sorted({k for r in all_runs for k in r.get("set", {})})
    write_tsv(out_root / "summary.tsv", rows, cols)
    print(f"\n{'run':<14}{'valid WER':>10}")
    for r in sorted(rows, key=lambda r: (r["valid_wer"] is None, r["valid_wer"])):
        print(f"{r['name']:<14}{r['valid_wer']:>10.4f}" if r["valid_wer"] is not None else f"{r['name']:<14}{'-':>10}")
    print(f"table: {out_root / 'summary.tsv'}")


if __name__ == "__main__":
    main()
