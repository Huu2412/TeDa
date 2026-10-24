"""End-to-end check of prepare_data + scan_text on a tiny synthetic ViMD-like parquet set."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from teda.data import audio as audio_lib
from teda.data.prepare import parse_shard_name, prepare
from teda.utils.manifest import read_tsv

ROOT = Path(__file__).resolve().parents[1]


def _numpy_resample(wave, sr, target_sr):
    """Stand-in for the torchaudio resampler so the plumbing is testable without torch.
    The real torchaudio path is checked by tests/test_audio.py (runs where torch is installed)."""
    n_new = int(round(wave.shape[1] * target_sr / sr))
    x_new = np.linspace(0, wave.shape[1] - 1, n_new)
    return np.interp(x_new, np.arange(wave.shape[1]), wave[0])[None, :].astype(np.float32)


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    monkeypatch.setattr(audio_lib, "_resample", _numpy_resample)


def _wav_bytes(sr=44100, ch=2, seconds=1.0, freq=440.0):
    t = np.arange(int(sr * seconds)) / sr
    left = 0.3 * np.sin(2 * np.pi * freq * t)
    right = 0.1 * np.sin(2 * np.pi * 2 * freq * t)
    data = np.stack([left, right][:ch], axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _make_shard(path, split_prefix, items):
    audio = [{"bytes": _wav_bytes(), "path": it["filename"]} for it in items]
    table = pa.table({
        "region": [it["region"] for it in items],
        "province_code": [it["pc"] for it in items],
        "province_name": ["X"] * len(items),
        "filename": [it["filename"] for it in items],
        "text": [it["text"] for it in items],
        "speakerID": ["spk"] * len(items),
        "gender": [0] * len(items),
        "audio": audio,
    })
    pq.write_table(table, path)


@pytest.fixture()
def shards(tmp_path):
    d = tmp_path / "src" / "data"
    d.mkdir(parents=True)
    _make_shard(d / "train-00000-of-00001.parquet", "train", [
        {"filename": "11_0001.wav", "region": "North", "pc": 11, "text": "Hôm nay ngày 2/9/1945, có 25 người."},
        {"filename": "21_0001.wav", "region": "Central", "pc": 21, "text": "Gọi UBND lúc 8h30 nhé"},
    ])
    _make_shard(d / "validation-00000-of-00001.parquet", "valid", [
        {"filename": "31_0001.wav", "region": "South", "pc": 31, "text": "Giá 3,5 triệu, tăng 50%."},
    ])
    _make_shard(d / "test-00000-of-00001.parquet", "test", [
        {"filename": "11_0002.wav", "region": "North", "pc": 11, "text": "Nghiên cứu học tập,các ứng dụng."},
    ])
    return tmp_path / "src"


def test_parse_shard_name():
    assert parse_shard_name("data/train-00007-of-00103.parquet") == ("train", 7, 103)
    assert parse_shard_name("data/validation-00000-of-00002.parquet")[0] == "valid"
    assert parse_shard_name("data/README.md") is None


def test_convert_to_16k_mono_uses_channel0_and_resamples():
    pcm, sr, ch = audio_lib.convert_to_16k_mono(_wav_bytes(44100, 2, 1.0))
    assert (sr, ch) == (44100, 2)
    assert pcm.dtype == np.int16 and abs(len(pcm) - 16000) <= 2
    # channel 0 is a 0.3-amplitude sine, channel 1 only 0.1 -> amplitude identifies the channel
    assert 0.25 * 32768 < np.abs(pcm).max() < 0.35 * 32768


def test_prepare_resume_and_scan(shards, tmp_path):
    out = tmp_path / "out"
    summary = prepare(out, local_shards=shards, workers=1)
    assert summary["n_utts"] == 4 and summary["n_failed"] == 0
    assert set(summary["splits"]) == {"train", "valid", "test"}

    for r in read_tsv(out / "manifest_raw.tsv"):
        info = sf.info(out / r["wav"])
        assert info.samplerate == 16000 and info.channels == 1
        assert abs(float(r["duration"]) - info.duration) < 1e-3

    # second run must skip finished shards and still give the same manifest
    before = (out / "manifest_raw.tsv").read_bytes()
    prepare(out, local_shards=shards, workers=1)
    assert (out / "manifest_raw.tsv").read_bytes() == before

    scan_out = tmp_path / "scan"
    final = tmp_path / "manifest.tsv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_text.py"), "--manifest-raw", str(out / "manifest_raw.tsv"),
         "--out-dir", str(scan_out), "--manifest-out", str(final), "--config", str(ROOT / "configs" / "text_norm.yaml")],
        check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
    )
    rows = {r["utt_id"]: r for r in read_tsv(final)}
    assert len(rows) == 4  # nothing dropped
    assert rows["11_0001"]["text_norm"] == (
        "hôm nay ngày hai tháng chín năm một nghìn chín trăm bốn mươi lăm có hai mươi lăm người"
    )
    assert rows["11_0001"]["region_id"] == "0" and rows["21_0001"]["region_id"] == "1" and rows["31_0001"]["region_id"] == "2"
    assert rows["11_0002"]["text_clean"] == "Nghiên cứu học tập, các ứng dụng."
    assert rows["31_0001"]["text_norm"] == "giá ba phẩy năm triệu tăng năm mươi phần trăm"
    assert rows["21_0001"]["has_digit"] == "1"

    stats = json.loads((scan_out / "stats.json").read_text(encoding="utf-8"))
    assert stats["splits"]["train"]["with_digit"] == 2
    unresolved = (scan_out / "unresolved.tsv").read_text(encoding="utf-8")
    assert "UBND" in unresolved  # acronym surfaced for review
    assert "11_0001" in (scan_out / "review_sample.tsv").read_text(encoding="utf-8")
    assert "int" in (scan_out / "norm_report.tsv").read_text(encoding="utf-8")
