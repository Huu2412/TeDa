"""Checks the REAL torchaudio resampling path (runs on Kaggle; skipped where torch is missing)."""

import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from teda.data import audio as audio_lib  # noqa: E402


def _wav(seconds=1.0, sr=44100, channels=2, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([x] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.mark.parametrize("sr", [44100, 48000, 22050])
def test_resample_to_16k_mono(sr):
    pcm, orig_sr, channels = audio_lib.convert_to_16k_mono(_wav(sr=sr))
    assert orig_sr == sr and channels == 2
    assert pcm.dtype == np.int16 and pcm.ndim == 1
    assert abs(len(pcm) - 16000) <= 2


def test_resampled_tone_keeps_frequency_and_amplitude():
    pcm, _, _ = audio_lib.convert_to_16k_mono(_wav(freq=1000.0))
    x = pcm.astype(np.float32) / 32768.0
    spec = np.abs(np.fft.rfft(x))
    peak_hz = np.argmax(spec) * 16000 / len(x)
    assert abs(peak_hz - 1000.0) < 5.0
    assert 0.25 < np.max(np.abs(x)) < 0.35


def test_uses_channel_zero_like_vipvl_pipeline():
    t = np.arange(44100) / 44100
    left = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = np.zeros_like(left)
    buf = io.BytesIO()
    sf.write(buf, np.stack([left, right], axis=1), 44100, format="WAV", subtype="PCM_16")
    pcm, _, _ = audio_lib.convert_to_16k_mono(buf.getvalue())
    assert np.max(np.abs(pcm)) > 5000  # left channel kept, not averaged with the silent right one
