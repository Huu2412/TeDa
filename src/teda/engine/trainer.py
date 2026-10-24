"""DDP trainer: fp16 AMP, gradient accumulation, resumable (last.pt), best.pt by valid WER, time limit.

Resume granularity: last.pt is written at optimizer-step boundaries and at epoch ends. A resumed run
re-creates the same global batch order for the current epoch (seed + epoch) and skips the batches already
done, so an interrupted Kaggle session continues where it stopped.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from teda.data.collate import collate, to_device
from teda.data.sampler import DynamicBatchSampler
from teda.engine.evaluator import evaluate
from teda.utils import dist as D
from teda.utils.ckpt import atomic_save, load_ckpt, read_json, write_json
from teda.utils.config import to_plain
from teda.utils.logging import MetricsWriter

log = logging.getLogger("teda")


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)  # one fbank thread per worker; avoids CPU oversubscription on Kaggle's few cores


def make_lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = min(1.0, (step - warmup) / max(total - warmup, 1))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return f


class Trainer:
    def __init__(self, cfg, model, train_ds, valid_ds, tokenizer, ctx: D.DistContext, out_dir: str | Path):
        self.cfg, self.tc, self.ctx = cfg, cfg.train, ctx
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tok, self.valid_ds = tokenizer, valid_ds
        self.raw = model.to(ctx.device)
        self.use_amp = str(self.tc.get("amp", "fp16")) == "fp16" and ctx.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ddp = (
            DDP(self.raw, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                find_unused_parameters=bool(self.tc.get("ddp_find_unused", False)))
            if ctx.distributed else self.raw
        )

        self.sampler = DynamicBatchSampler(
            train_ds.num_frames, int(self.tc.max_frames_per_batch), ctx.world_size, ctx.rank,
            shuffle=True, seed=int(self.tc.seed), max_batch_size=self.tc.get("max_batch_size"),
        )
        nw = int(self.tc.get("num_workers", 2))
        pin = bool(self.tc.get("pin_memory", False))
        self.loader = DataLoader(
            train_ds, batch_sampler=self.sampler, collate_fn=collate, num_workers=nw,
            pin_memory=pin, persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None, worker_init_fn=_worker_init,
        )
        self.accum = int(self.tc.get("accum_grad", 1))
        self.micro_per_epoch = len(self.sampler)
        self.steps_per_epoch = math.ceil(self.micro_per_epoch / self.accum)
        self.total_steps = self.steps_per_epoch * int(self.tc.epochs)
        log.info("train: %d utts, %d micro-batches/epoch/rank, %d optimizer steps/epoch, %d total steps "
                 "(effective batch ~%d utts)", len(train_ds), self.micro_per_epoch, self.steps_per_epoch,
                 self.total_steps, round(len(train_ds) / self.steps_per_epoch))

        self.opt = self._build_optimizer()
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, make_lr_lambda(int(self.tc.warmup_steps), self.total_steps, float(self.tc.get("min_lr_ratio", 0.05))))
        self.metrics = MetricsWriter(self.out / "metrics.jsonl", enabled=ctx.is_main)
        self.step = 0
        self.epoch = 0
        self.micro_in_epoch = 0
        self.best_wer = float("inf")
        self.best_step = -1

    # ------------------------------------------------------------------------ setup
    def _build_optimizer(self):
        groups = []
        for name, lr in (("encoder", float(self.tc.lr_encoder)), ("head", float(self.tc.lr_head))):
            decay, no_decay = [], []
            for n, p in self.raw.named_parameters():
                if not p.requires_grad or (n.startswith("encoder.") != (name == "encoder")):
                    continue
                (no_decay if p.ndim <= 1 else decay).append(p)
            wd = float(self.tc.get("weight_decay", 0.01))
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": wd, "name": f"{name}_decay"})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{name}_no_decay"})
        return torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)

    # --------------------------------------------------------------------- checkpoints
    def _state(self) -> Dict:
        return {
            "model": self.raw.state_dict(), "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(), "epoch": self.epoch, "micro_in_epoch": self.micro_in_epoch,
            "step": self.step, "best_wer": self.best_wer, "best_step": self.best_step,
        }

    def save_last(self) -> None:
        if self.ctx.is_main:
            atomic_save(self._state(), self.out / "last.pt")

    def _maybe_resume(self) -> None:
        path = self.out / "last.pt"
        if not path.exists():
            return
        st = load_ckpt(path)
        self.raw.load_state_dict(st["model"])
        try:
            self.opt.load_state_dict(st["optimizer"])
        except ValueError as e:
            raise RuntimeError(
                f"{path}: optimizer state does not match this model. Did the freeze settings change since "
                f"the checkpoint was written? Use a new out-dir or delete last.pt. ({e})") from e
        self.sched.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        self.epoch, self.micro_in_epoch = int(st["epoch"]), int(st["micro_in_epoch"])
        self.step, self.best_wer, self.best_step = int(st["step"]), float(st["best_wer"]), int(st["best_step"])
        log.info("resumed from %s: epoch %d, micro-batch %d, step %d, best valid WER %.4f",
                 path, self.epoch, self.micro_in_epoch, self.step, self.best_wer)

    # ------------------------------------------------------------------------- loop
    def _time_exceeded(self, t0: float) -> bool:
        limit = self.tc.get("max_hours")
        flag = bool(limit) and (time.time() - t0) > float(limit) * 3600
        return bool(D.broadcast_object(flag))  # rank 0 decides, so every rank stops at the same step

    def fit(self) -> Dict:
        done = read_json(self.out / "done.json")
        if done and not self.tc.get("force", False):
            log.info("already finished (%s); nothing to do. Delete done.json or set train.force=true to rerun.", self.out / "done.json")
            return done
        self._maybe_resume()
        t0 = time.time()
        self.raw.train()
        win_loss, win_n, win_gn, nonfinite = 0.0, 0, 0.0, 0
        log_every = int(self.tc.get("log_every", 50))
        save_every = int(self.tc.get("save_every_steps", 500))
        status = "done"

        for epoch in range(self.epoch, int(self.tc.epochs)):
            self.epoch = epoch
            self.sampler.set_epoch(epoch, skip=self.micro_in_epoch)
            # The number of batches differs slightly between epochs (shuffled bucketing), so take the exact
            # count for THIS epoch; self.micro_per_epoch is only the epoch-0 estimate used for the LR schedule.
            n_micro = len(self.sampler)
            accum_n = 0
            for batch in self.loader:
                self.micro_in_epoch += 1
                accum_n += 1
                last_micro = self.micro_in_epoch >= n_micro
                boundary = accum_n == self.accum or last_micro
                inputs = to_device({k: batch[k] for k in ("feats", "feat_lens", "tokens", "token_lens")}, self.ctx.device)

                sync = self.ddp.no_sync() if (self.ctx.distributed and not boundary) else contextlib.nullcontext()
                with sync:
                    with torch.autocast(device_type=self.ctx.device.type, dtype=torch.float16, enabled=self.use_amp):
                        out = self.ddp(inputs)
                    loss = out["loss"]
                    if not torch.isfinite(loss):
                        nonfinite += 1
                    self.scaler.scale(loss / self.accum).backward()
                win_loss += float(out["loss_ctc"])
                win_n += 1
                if not boundary:
                    continue

                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], float(self.tc.grad_clip))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                accum_n = 0
                win_gn += float(gn)

                if self.step % log_every == 0:
                    avg = D.all_reduce_mean(win_loss / max(win_n, 1), self.ctx.device)
                    mem = torch.cuda.max_memory_allocated() / 2**30 if self.ctx.device.type == "cuda" else 0.0
                    log.info("ep %d step %d/%d | ctc %.3f | lr %.2e | gnorm %.2f | %.1f GiB | nonfinite %d | %.0f min",
                             epoch, self.step, self.total_steps, avg, self.sched.get_last_lr()[0], win_gn / log_every,
                             mem, nonfinite, (time.time() - t0) / 60)
                    self.metrics.write(kind="train", step=self.step, epoch=epoch, loss_ctc=avg,
                                       lr=self.sched.get_last_lr()[0], grad_norm=win_gn / log_every, peak_gib=mem)
                    win_loss, win_n, win_gn = 0.0, 0, 0.0
                if self.step % save_every == 0:
                    self.save_last()
                if self.step % int(self.tc.get("time_check_every", 20)) == 0 and self._time_exceeded(t0):
                    self.save_last()
                    status = "time_limit"
                    log.warning("time limit (%.1f h) reached at step %d: saved last.pt. Re-run the same command to resume.",
                                float(self.tc.max_hours), self.step)
                    return {"status": status, "step": self.step, "best_wer": self.best_wer}

            # ---- end of epoch
            self.epoch, self.micro_in_epoch = epoch + 1, 0
            if (epoch + 1) % int(self.tc.get("eval_every_epochs", 1)) == 0 or epoch + 1 == int(self.tc.epochs):
                self.validate()
            self.save_last()
            import gc
            gc.collect()
            if self.ctx.device.type == "cuda":
                torch.cuda.empty_cache()

        result = {"status": status, "step": self.step, "epochs": int(self.tc.epochs),
                  "best_valid_wer": self.best_wer, "best_step": self.best_step}
        if self.ctx.is_main:
            write_json(self.out / "done.json", result)
        log.info("finished: best valid WER %.4f at step %d", self.best_wer, self.best_step)
        return result

    # -------------------------------------------------------------------- validation
    def validate(self) -> Dict:
        ec = self.cfg.get("eval", {})
        summary, _ = evaluate(
            self.raw, self.valid_ds, self.tok, self.ctx, int(ec.get("batch_frames", 20000)),
            amp=self.use_amp, chunk=ec.get("chunk"), num_workers=int(self.tc.get("num_workers", 0)),
            pin_memory=bool(self.tc.get("pin_memory", False)),
            n_boot=0 if not ec.get("bootstrap_in_training", False) else 1000,
        )
        wer = summary["wer"]
        reg = " ".join(f"{k}={v['wer']:.4f}" for k, v in summary["by_region"].items())
        log.info("VALID epoch %d step %d | WER %.4f CER %.4f | %s", self.epoch, self.step, wer, summary["cer"], reg)
        self.metrics.write(kind="valid", step=self.step, epoch=self.epoch, wer=wer, cer=summary["cer"],
                           by_region={k: v["wer"] for k, v in summary["by_region"].items()})
        if wer < self.best_wer:
            self.best_wer, self.best_step = wer, self.step
            if self.ctx.is_main:
                atomic_save({"model": self.raw.state_dict(), "step": self.step, "epoch": self.epoch, "valid_wer": wer,
                             "cfg": to_plain(self.cfg)}, self.out / "best.pt")
            log.info("new best valid WER %.4f (saved best.pt)", wer)
        self.raw.train()
        import gc
        gc.collect()
        if self.ctx.device.type == "cuda":
            torch.cuda.empty_cache()
        return summary
