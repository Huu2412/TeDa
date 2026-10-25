"""One-time conversion of ViMD (HF parquet shards) to 16 kHz mono wav + raw manifest.

Audio handling (teda/data/audio.py) mirrors the ChunkFormer / ViP-VL data pipeline
(chunkformer/dataset/processor.py): decode -> keep channel 0 -> torchaudio
``Resample`` to 16 kHz. The model computes fbank + global CMVN itself later.

Shards are processed one at a time (download -> convert -> delete) so the ~60 GB
source never has to fit on disk, and finished shards are skipped on re-run.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from teda.data import audio as audio_lib
from teda.utils.manifest import RAW_FIELDS, read_tsv, write_tsv

SPLIT_ALIASES = {"train": "train", "test": "test", "valid": "valid", "validation": "valid", "val": "valid", "dev": "valid"}
_SHARD_RE = re.compile(r"(?:^|[/\\])(?P<split>[A-Za-z]+)-(?P<idx>\d+)-of-(?P<n>\d+)\.parquet$")
_META_COLS = ["filename", "text", "region", "province_code", "province_name", "gender"]


# ------------------------------------------------------------------------- audio
# Conversion itself lives in teda/data/audio.py (decode -> channel 0 -> torchaudio Resample).
def _convert_task(task: Tuple[str, bytes, str]) -> Dict:
    utt_id, audio_bytes, out_path = task
    try:
        pcm, sr, n_ch = audio_lib.convert_to_16k_mono(audio_bytes)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        audio_lib.write_wav_16k(out_path, pcm)
        return {"utt_id": utt_id, "ok": True, "duration": len(pcm) / audio_lib.TARGET_SR, "orig_sr": sr, "channels": n_ch}
    except Exception as e:  # keep going, report at the end
        return {"utt_id": utt_id, "ok": False, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------ shards
def parse_shard_name(path: str) -> Optional[Tuple[str, int, int]]:
    m = _SHARD_RE.search(path)
    if not m or m.group("split").lower() not in SPLIT_ALIASES:
        return None
    return SPLIT_ALIASES[m.group("split").lower()], int(m.group("idx")), int(m.group("n"))


def list_remote_shards(repo_id: str) -> List[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and parse_shard_name(f))


def list_local_shards(local_dir: str | Path) -> List[str]:
    root = Path(local_dir)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.parquet") if parse_shard_name(p.name))


def _download(repo_id: str, remote: str, cache_dir: Path, retries: int = 3) -> Path:
    from huggingface_hub import hf_hub_download

    for attempt in range(1, retries + 1):
        try:
            return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=remote, local_dir=str(cache_dir)))
        except Exception as e:
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"  download failed ({type(e).__name__}: {e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _iter_parquet_rows(path: Path, batch_size: int = 64) -> Iterable[Dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols = [c for c in _META_COLS + ["audio"] if c in pf.schema_arrow.names]
    if "audio" not in cols or "text" not in cols:
        raise ValueError(f"{path}: expected 'audio' and 'text' columns, found {pf.schema_arrow.names}")
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield from batch.to_pylist()


def process_shard(
    parquet_path: Path,
    split: str,
    out_dir: Path,
    workers: int = 1,
    pool: Optional["mp.pool.Pool"] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Convert every utterance of one parquet shard. Returns (manifest rows, failures)."""
    rows: List[Dict] = []
    failures: List[Dict] = []

    def flush(batch_meta: List[Dict], tasks: List[Tuple]) -> None:
        if not tasks:
            return
        results = pool.imap(_convert_task, tasks, chunksize=4) if pool is not None else map(_convert_task, tasks)
        for meta, res in zip(batch_meta, results):
            if res["ok"]:
                meta = dict(meta, duration=f"{res['duration']:.3f}", orig_sr=res["orig_sr"])
                rows.append(meta)
            else:
                failures.append({"utt_id": meta["utt_id"], "split": split, "error": res["error"]})

    batch_meta: List[Dict] = []
    tasks: List[Tuple] = []
    for r in _iter_parquet_rows(parquet_path):
        audio = r.get("audio") or {}
        audio_bytes = audio.get("bytes")
        fname = r.get("filename") or audio.get("path") or ""
        utt_id = Path(str(fname)).stem
        if not utt_id or not audio_bytes:
            failures.append({"utt_id": utt_id or "?", "split": split, "error": "missing filename or audio bytes"})
            continue
        rel = f"wav16k/{split}/{utt_id}.wav"
        batch_meta.append({
            "utt_id": utt_id, "split": split, "wav": rel, "text_raw": r.get("text", ""),
            "region": r.get("region", ""), "province_code": r.get("province_code", ""),
            "province_name": r.get("province_name", ""), "gender": r.get("gender", ""),
        })
        tasks.append((utt_id, audio_bytes, str(out_dir / rel)))
        if len(tasks) >= 128:
            flush(batch_meta, tasks)
            batch_meta, tasks = [], []
    flush(batch_meta, tasks)
    return rows, failures


# ------------------------------------------------------------------------- driver
def prepare(
    out_dir: str | Path,
    repo_id: str = "nguyendv02/ViMD_Dataset",
    local_shards: Optional[str | Path] = None,
    splits: Iterable[str] = ("train", "valid", "test"),
    workers: int = 2,
    limit_shards: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    keep_downloads: bool = False,
) -> Dict:
    out_dir = Path(out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else out_dir / "_download"
    want = {SPLIT_ALIASES[s] for s in splits}

    remote = list_local_shards(local_shards) if local_shards else list_remote_shards(repo_id)
    plan = [(f, parsed[0]) for f in remote if (parsed := parse_shard_name(f)) is not None and parsed[0] in want]
    if limit_shards:
        counts: Counter = Counter()
        limited = []
        for f, sp in plan:
            if counts[sp] < limit_shards:
                limited.append((f, sp))
                counts[sp] += 1
        plan = limited
    print(f"{len(plan)} shards to process: " + ", ".join(f"{sp}={n}" for sp, n in Counter(sp for _, sp in plan).items()), flush=True)

    pool = mp.Pool(workers, initializer=audio_lib.init_worker) if workers > 1 else None
    all_failures: List[Dict] = []
    try:
        for i, (rel_path, split) in enumerate(plan, 1):
            stem = Path(rel_path).stem
            marker = shard_dir / f"{stem}.tsv"
            if marker.exists():
                print(f"[{i}/{len(plan)}] {stem}: done, skipping", flush=True)
                continue
            t0 = time.time()
            if local_shards:
                src, downloaded = Path(local_shards) / rel_path, False
            else:
                src, downloaded = _download(repo_id, rel_path, cache_dir), True
            rows, failures = process_shard(src, split, out_dir, workers, pool)
            write_tsv(marker, rows, RAW_FIELDS)  # written last => marks the shard as finished
            if failures:
                with open(shard_dir / f"{stem}.failures.json", "w", encoding="utf-8") as f:
                    json.dump(failures, f, ensure_ascii=False, indent=1)
            all_failures += failures
            if downloaded and not keep_downloads:
                src.unlink(missing_ok=True)
            print(f"[{i}/{len(plan)}] {stem}: {len(rows)} ok, {len(failures)} failed, {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return merge_manifest(out_dir)


def merge_manifest(out_dir: str | Path) -> Dict:
    """Merge per-shard TSVs into manifest_raw.tsv and write prepare_summary.json."""
    out_dir = Path(out_dir)
    rows: List[Dict] = []
    for p in sorted((out_dir / "shards").glob("*.tsv")):
        rows += read_tsv(p)
    seen: Dict[str, str] = {}
    dups = []
    for r in rows:
        if r["utt_id"] in seen:
            dups.append(r["utt_id"])
        seen[r["utt_id"]] = r["split"]
    if dups:
        raise ValueError(f"{len(dups)} duplicate utt_id values, e.g. {dups[:5]}; cannot build a unique manifest")
    rows.sort(key=lambda r: (r["split"], r["utt_id"]))
    write_tsv(out_dir / "manifest_raw.tsv", rows, RAW_FIELDS)

    summary: Dict = {"n_utts": len(rows), "splits": {}, "orig_sr": dict(Counter(r["orig_sr"] for r in rows))}
    for sp in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == sp]
        summary["splits"][sp] = {
            "n_utts": len(sub),
            "hours": round(sum(float(r["duration"]) for r in sub) / 3600, 3),
            "regions": dict(Counter(r["region"] for r in sub)),
        }
    n_fail = sum(len(json.load(open(p, encoding="utf-8"))) for p in (out_dir / "shards").glob("*.failures.json"))
    summary["n_failed"] = n_fail
    with open(out_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
