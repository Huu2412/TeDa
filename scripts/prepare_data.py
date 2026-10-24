#!/usr/bin/env python
"""Step 0a: convert ViMD to 16 kHz mono wav and write <out-dir>/manifest_raw.tsv.

Kaggle example:
    python scripts/prepare_data.py --out-dir /kaggle/working/vimd16k --workers 4

Re-run the same command after an interrupted session: finished shards are skipped.
Use --list to inspect the shards first, --limit-shards 1 for a quick dry run.
"""

import argparse
import os

from teda.data.prepare import SPLIT_ALIASES, list_local_shards, list_remote_shards, parse_shard_name, prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="where wav16k/, shards/ and manifest_raw.tsv go")
    ap.add_argument("--repo-id", default="nguyendv02/ViMD_Dataset")
    ap.add_argument("--local-shards", default=None, help="use parquet shards already on disk instead of downloading")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=sorted(SPLIT_ALIASES))
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--limit-shards", type=int, default=None, help="only the first N shards per split (dry run)")
    ap.add_argument("--cache-dir", default=None, help="temporary download folder (default <out-dir>/_download)")
    ap.add_argument("--keep-downloads", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the shards found and exit")
    a = ap.parse_args()

    if a.list:
        files = list_local_shards(a.local_shards) if a.local_shards else list_remote_shards(a.repo_id)
        for f in files:
            print(f, parse_shard_name(f)[0])
        print(f"{len(files)} shards")
        return

    prepare(
        out_dir=a.out_dir, repo_id=a.repo_id, local_shards=a.local_shards, splits=a.splits,
        workers=a.workers, limit_shards=a.limit_shards,
        cache_dir=a.cache_dir, keep_downloads=a.keep_downloads,
    )


if __name__ == "__main__":
    main()
