#!/usr/bin/env python
"""Find a safe `train.max_frames_per_batch` for your GPU, using synthetic batches (no dataset needed).

    python scripts/probe_memory.py --exp e00_baseline --frames 12000 18000 24000 32000

For each value it runs a few real forward/backward/optimizer steps (fp16, current freeze settings)
on a batch of n utterances of --utt-seconds each (n * length ~ frames), then prints peak GPU memory
and throughput. Pick the largest value that leaves a few GiB free: real batches vary in length and
the first epoch is when out-of-memory errors tend to appear. Run it on ONE T4 (per-GPU memory is what matters).
"""

import argparse
import time

import torch

from teda.data.features import subsampled_length
from teda.models.teda_model import build_model
from teda.utils.config import load_config
from teda.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="e00_baseline")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--frames", type=int, nargs="+", default=[12000, 18000, 24000, 32000])
    ap.add_argument("--utt-seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device: this probe is meant for the Kaggle GPU.")

    setup_logging(0)
    cfg = load_config(a.config, a.exp, a.set)
    dev = torch.device("cuda", 0)
    model, _ = build_model(cfg, int(cfg.data.vocab_size))
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gib:.1f} GiB), trainable params: {sum(p.numel() for p in params):,}")

    T = int(a.utt_seconds * 100)
    U = min(int(7 * a.utt_seconds), subsampled_length(T) // 2)  # ~7 BPE tokens/s, always CTC-feasible
    for frames in a.frames:
        n = max(1, frames // T)
        batch = {"feats": torch.randn(n, T, 80, device=dev), "feat_lens": torch.full((n,), T, device=dev),
                 "tokens": torch.randint(2, 1024, (n, U), device=dev), "token_lens": torch.full((n,), U, device=dev)}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for s in range(a.steps):
                if s == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(batch)["loss"]
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            sec = (time.time() - t0) / max(a.steps - 1, 1)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"frames={frames:>6} ({n} x {a.utt_seconds:.0f}s = {n * a.utt_seconds:.0f}s audio): peak {peak:5.1f} GiB, "
                  f"{sec:5.2f} s/step, {n * a.utt_seconds / sec:6.0f} audio-s/s")
        except torch.cuda.OutOfMemoryError:
            print(f"frames={frames:>6}: OUT OF MEMORY; stop here and use a smaller value")
            break
        finally:
            del batch
    print("Note: dynamic chunking picks a random context per step, so peak memory varies; leave headroom.")


if __name__ == "__main__":
    main()
