"""Audio conversion to 16 kHz mono, mirroring the ChunkFormer / ViP-VL data pipeline.

The ViP-VL pipeline (chunkformer/dataset/processor.py) does, per sample:
    waveform = torchaudio.load(...)                      # float32 in [-1, 1]
    waveform = waveform[0:1]  if multi-channel           # ``singal_channel`` with channel=0
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
We do exactly the same, once, offline, and store 16-bit PCM wav. The only difference is
the final rounding to int16 (quantisation noise ~ -96 dB), which is negligible.

torch / torchaudio are imported lazily so that the rest of ``teda.data`` stays import-light.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Tuple

import numpy as np

TARGET_SR = 16000


def decode_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio bytes -> (waveform float32 of shape (channels, samples), sample rate)."""
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)  # (samples, channels)
    return np.ascontiguousarray(x.T), int(sr)


def select_channel(wave: np.ndarray, channel: int = 0) -> np.ndarray:
    """Keep one channel (default 0, as the ViP-VL pipeline does). Shape stays (1, samples)."""
    if wave.shape[0] != 1:
        if channel >= wave.shape[0]:
            raise ValueError(f"channel {channel} not in {wave.shape[0]}-channel audio")
        wave = wave[channel : channel + 1]
    return wave


@lru_cache(maxsize=8)
def _get_resampler(orig_sr: int, target_sr: int):
    import torchaudio

    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)


def _resample(wave: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """(1, T) float32 at ``sr`` -> (1, T') float32 at ``target_sr`` with torchaudio."""
    import torch

    resampler = _get_resampler(sr, target_sr)
    with torch.no_grad():
        out = resampler(torch.from_numpy(wave))
    return out.numpy()


def to_int16(wave: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(wave * 32768.0), -32768, 32767).astype(np.int16)


def convert_to_16k_mono(data: bytes, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int, int]:
    """Bytes of any audio file -> (int16 mono samples at 16 kHz, original sr, original channels)."""
    wave, sr = decode_wav_bytes(data)
    channels = wave.shape[0]
    wave = select_channel(wave, 0)
    if sr != target_sr:
        wave = _resample(wave, sr, target_sr)
    return to_int16(wave[0]), sr, channels


def write_wav_16k(path: str, pcm: np.ndarray, sr: int = TARGET_SR) -> None:
    import soundfile as sf

    sf.write(path, pcm, sr, subtype="PCM_16", format="WAV")


def init_worker() -> None:
    """Worker initializer: one torch thread per process to avoid CPU oversubscription."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # torch not installed (e.g. unit tests) - fine
        pass
