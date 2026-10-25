"""Waveform -> 80-dim log-mel fbank, exactly like the ChunkFormer / ViP-VL pipeline.

Pipeline reference (chunkformer/dataset/processor.py::compute_fbank):
    waveform * 2**15  ->  torchaudio.compliance.kaldi.fbank(num_mel_bins=80, frame_length=25,
    frame_shift=10, dither, energy_floor=0, window_type="povey")
Global CMVN is applied inside the encoder (it owns the `global_cmvn` module), not here.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch

SAMPLE_RATE = 16000
FRAME_LEN = 400   # 25 ms
FRAME_SHIFT = 160  # 10 ms


def num_fbank_frames(n_samples: int) -> int:
    """Number of kaldi fbank frames (snip_edges=True) for n_samples at 16 kHz."""
    return 0 if n_samples < FRAME_LEN else 1 + (n_samples - FRAME_LEN) // FRAME_SHIFT


def subsampled_length(n_frames: int) -> int:
    """Encoder output length for n_frames input frames.

    DepthwiseConvSubsampling = three convs with kernel 3, stride 2, padding 0 (subsampling rate 8),
    so each stage maps n -> (n - 3) // 2 + 1. Checked against the real module in tests/test_model.py.
    """
    n = n_frames
    for _ in range(3):
        if n < 3:
            return 0
        n = (n - 3) // 2 + 1
    return n


def speed_perturb(wave: torch.Tensor, speeds: Sequence[float], sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Kaldi-style speed perturbation: resample so that playback at `sr` is faster/slower."""
    if not speeds:
        return wave
    s = random.choice(list(speeds))
    if abs(s - 1.0) < 1e-6:
        return wave
    import torchaudio.functional as AF

    return AF.resample(wave, orig_freq=int(round(sr * s)), new_freq=sr)


def compute_fbank(wave: torch.Tensor, num_mel_bins: int = 80, dither: float = 0.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """wave: float tensor in [-1, 1], shape (samples,) or (1, samples). Returns (frames, num_mel_bins)."""
    import torchaudio.compliance.kaldi as kaldi

    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    return kaldi.fbank(
        wave * (1 << 15),
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=sr,
        window_type="povey",
    )


def spec_aug(
    feat: torch.Tensor,
    num_t_mask: int = 2,
    num_f_mask: int = 2,
    max_t: int = 50,
    max_f: int = 10,
) -> torch.Tensor:
    """Same SpecAugment (time/frequency masking, no warp) as the ChunkFormer recipe. Returns a copy."""
    y = feat.clone()
    n_frames, n_freq = y.shape
    for _ in range(num_t_mask):
        start = random.randint(0, max(n_frames - 1, 0))
        length = random.randint(1, max_t)
        y[start : min(n_frames, start + length), :] = 0
    for _ in range(num_f_mask):
        start = random.randint(0, n_freq - 1)
        length = random.randint(1, max_f)
        y[:, start : min(n_freq, start + length)] = 0
    return y


def feature_cfg_to_kwargs(fcfg: Dict) -> Dict:
    return {"num_mel_bins": int(fcfg.get("num_mel_bins", 80))}
