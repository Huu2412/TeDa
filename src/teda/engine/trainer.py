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


def _host_ram_gib() -> float:
    """Return current process RSS (resident set size) in GiB. Falls back to 0 if psutil is absent."""
    try:
        import psutil
        import os
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
    except Exception:
        return 0.0


def _sys_ram_gib() -> float:
    """Return total system used RAM in GiB."""
    try:
        import psutil
        return psutil.virtual_memory().used / 2 ** 30
    except Exception:
        return 0.0


def _malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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

        self.pad_multiple = int(self.tc.get("pad_multiple", 1))
        self.sampler = DynamicBatchSampler(
            train_ds.num_frames, int(self.tc.max_frames_per_batch), ctx.world_size, ctx.rank,
            shuffle=True, seed=int(self.tc.seed), max_batch_size=self.tc.get("max_batch_size"),
            pad_multiple=self.pad_multiple,
        )
        nw = int(self.tc.get("num_workers", 2))
        pin = bool(self.tc.get("pin_memory", False))
        pw = bool(self.tc.get("persistent_workers", True)) if nw > 0 else False
        pf = int(self.tc.get("prefetch_factor", 2)) if nw > 0 else None
        _collate = (lambda b: collate(b, self.pad_multiple)) if self.pad_multiple > 1 else collate
        self.loader = DataLoader(
            train_ds, batch_sampler=self.sampler, collate_fn=_collate, num_workers=nw,
            pin_memory=pin, persistent_workers=pw,
            prefetch_factor=pf, worker_init_fn=_worker_init,
        )
        # Limit the cuDNN autotuner plan cache: each unique tensor shape causes cuDNN to allocate
        # and hold a convolution plan in *host* RAM. With dynamic audio batching the number of
        # distinct shapes can be very large (especially after unfreezing all 12 blocks). Capping
        # the cache at 32 (per engine) prevents unbounded growth while still covering most shapes.
        if ctx.device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = False      # benchmark=True makes the problem worse
                torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
            except Exception:
                pass
            # torch >= 2.3 exposes a plan-cache limit; ignore silently on older versions.
            try:
                torch.cuda.set_per_process_memory_fraction(1.0)  # no-op but ensures driver is init
            except Exception:
                pass
        self.max_host_gib: Optional[float] = float(self.tc.max_host_gib) if self.tc.get("max_host_gib") else None
        self.accum = int(self.tc.get("accum_grad", 1))
        ac = (cfg.get("model", {}) or {}).get("accent", {}) or {}
        self.acc_start = int(ac.get("start_step", 0))      # before this step the accent head sees detached features
        self.acc_ramp = int(ac.get("ramp_steps", 0))       # then its loss weight grows 0 -> 1 over this many steps
        al = (cfg.get("model", {}) or {}).get("align", {}) or {}
        self.al_start = int(al.get("start_step", 0))        # before this step the aligner learns on detached features
        self.al_ramp = int(al.get("ramp_steps", 0))         # then its loss weight grows 0 -> 1 over this many steps
        self.probe_every = int(ac.get("grad_probe_every", 0))
        self.probe_block = int(ac.get("grad_probe_block", -1))
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
    def _grad_probe(self, out: Dict) -> Optional[Dict]:
        """How large is the accent gradient next to CTC's on ONE encoder block? Uses autograd.grad on the graph of
        this micro-batch (nothing is accumulated, no DDP communication, rank 0 only). Under fp16 both losses are
        multiplied by the GradScaler scale first so small gradients do not underflow."""
        params = [p for p in self.raw.encoder.net.encoders[self.probe_block].parameters() if p.requires_grad]
        if not params:
            return None
        scale = float(self.scaler.get_scale()) if self.use_amp else 1.0
        g_ctc = torch.autograd.grad(out["_graph_ctc"] * scale, params, retain_graph=True, allow_unused=True)
        g_acc = torch.autograd.grad(out["_graph_acc"] * scale, params, retain_graph=True, allow_unused=True)
        vec = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).float().flatten() for g, p in zip(gs, params)]) / scale
        a, b = vec(g_ctc), vec(g_acc)
        na, nb = float(a.norm()), float(b.norm())
        if not (math.isfinite(na) and math.isfinite(nb)):
            return None  # fp16 overflow in this probe; the next one will do
        return {"grad_ctc": na, "grad_acc": nb, "acc_share": nb / max(na + nb, 1e-12),
                "cos": float(torch.dot(a, b) / max(na * nb, 1e-12))}

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
        win_acc_loss, win_acc_acc = 0.0, 0.0
        win_al = dict.fromkeys(("align", "qty", "cos", "top1", "n"), 0.0)
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
                inputs = to_device({k: batch[k] for k in ("feats", "feat_lens", "tokens", "token_lens", "region_ids", "word_targets", "n_words") if k in batch}, self.ctx.device)

                if self.raw.accent is not None:
                    # phase 1 (step < start): head trains on detached features; phase 2: attached, weight ramps 0 -> 1
                    self.raw.accent_detached = self.step < self.acc_start
                    self.raw.aux_scale = min(1.0, (self.step - self.acc_start + 1) / self.acc_ramp) if self.acc_ramp else 1.0

                if self.raw.align is not None:
                    self.raw.align_detached = self.step < self.al_start
                    self.raw.align_scale = min(1.0, (self.step - self.al_start + 1) / self.al_ramp) if self.al_ramp else 1.0

                sync = self.ddp.no_sync() if (self.ctx.distributed and not boundary) else contextlib.nullcontext()
                with sync:
                    with torch.autocast(device_type=self.ctx.device.type, dtype=torch.float16, enabled=self.use_amp):
                        out = self.ddp(inputs)
                    loss = out["loss"]
                    if not torch.isfinite(loss):
                        nonfinite += 1
                    if (self.ctx.is_main and self.probe_every and "_graph_acc" in out and accum_n == 1
                            and (self.step + 1) % self.probe_every == 0):
                        pr = self._grad_probe(out)
                        if pr:
                            log.info("grad probe step %d (block %d): |g_ctc| %.3g |g_acc| %.3g -> accent share %.1f%% | cos(g_ctc,g_acc) %+.2f",
                                     self.step + 1, self.probe_block % len(self.raw.encoder.net.encoders), pr["grad_ctc"],
                                     pr["grad_acc"], 100 * pr["acc_share"], pr["cos"])
                            self.metrics.write(kind="grad_probe", step=self.step + 1, **pr)
                    self.scaler.scale(loss / self.accum).backward()
                win_loss += float(out["loss_ctc"])
                if "loss_accent" in out:
                    win_acc_loss += float(out["loss_accent"])
                    win_acc_acc += float(out["accent_acc"])
                if "loss_align" in out:
                    win_al["align"] += float(out["loss_align"])
                    win_al["qty"] += float(out["loss_qty"])
                    win_al["cos"] += float(out["align_cos"])
                    win_al["top1"] += float(out["align_top1"])
                    win_al["n"] += 1
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
                    gpu_gib = torch.cuda.max_memory_allocated() / 2**30 if self.ctx.device.type == "cuda" else 0.0
                    host_gib = _host_ram_gib()
                    sys_gib = _sys_ram_gib()
                    acc_str = f" | acc_loss {win_acc_loss/win_n:.3f} acc {100*win_acc_acc/win_n:.1f}%" if win_acc_loss > 0 else ""
                    log.info("ep %d step %d/%d | ctc %.3f%s | lr %.2e | gnorm %.2f | gpu %.1f GiB | host %.1f GiB | sys %.1f GiB | nonfinite %d | %.0f min",
                             epoch, self.step, self.total_steps, avg, acc_str, self.sched.get_last_lr()[0], win_gn / log_every,
                             gpu_gib, host_gib, sys_gib, nonfinite, (time.time() - t0) / 60)
                    metrics_kwargs = {}
                    if win_al["n"]:
                        k = win_al["n"]
                        al_str = (f" | align {win_al['align']/k:.3f} cos {win_al['cos']/k:.3f} top1 {100*win_al['top1']/k:.1f}%"
                                  f" qty {win_al['qty']/k:.3f}{' (warm-up, detached)' if self.raw.align_detached else ''}")
                        log.info("align step %d%s", self.step, al_str)
                        metrics_kwargs.update(loss_align=win_al["align"] / k, loss_qty=win_al["qty"] / k,
                                              align_cos=win_al["cos"] / k, align_top1=win_al["top1"] / k)
                    if win_acc_loss > 0:
                        metrics_kwargs["loss_accent"] = win_acc_loss / win_n
                        metrics_kwargs["accent_acc"] = win_acc_acc / win_n
                    self.metrics.write(kind="train", step=self.step, epoch=epoch, loss_ctc=avg,
                                       lr=self.sched.get_last_lr()[0], grad_norm=win_gn / log_every,
                                       peak_gpu_gib=gpu_gib, host_gib=host_gib, sys_gib=sys_gib, **metrics_kwargs)
                    win_loss, win_n, win_gn = 0.0, 0, 0.0
                    win_acc_loss, win_acc_acc = 0.0, 0.0
                    win_al = dict.fromkeys(("align", "qty", "cos", "top1", "n"), 0.0)
                    _malloc_trim()
                sys_gib = _sys_ram_gib()
                if (self.max_host_gib and _host_ram_gib() > self.max_host_gib) or (sys_gib > 27.0):
                    self.save_last()
                    status = "host_ram_limit"
                    log.warning("host RAM limit exceeded at step %d (process: %.1f GiB, sys: %.1f GiB): saved last.pt. Re-run the same command to resume.",
                                self.step, _host_ram_gib(), sys_gib)
                    return {"status": status, "step": self.step, "best_wer": self.best_wer}
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
            _malloc_trim()

        result = {"status": status, "step": self.step, "epochs": int(self.tc.epochs),
                  "best_valid_wer": self.best_wer, "best_step": self.best_step}
        if self.ctx.is_main:
            write_json(self.out / "done.json", result)
        log.info("finished: best valid WER %.4f at step %d", self.best_wer, self.best_step)
        return result

    # -------------------------------------------------------------------- validation
    def validate(self) -> Dict:
        ec = self.cfg.get("eval", {})
        pad_multiple = int(ec.get("pad_multiple", self.pad_multiple))
        _collate = (lambda b: collate(b, pad_multiple)) if pad_multiple > 1 else collate
        summary, recs = evaluate(
            self.raw, self.valid_ds, self.tok, self.ctx, int(ec.get("batch_frames", 20000)),
            amp=self.use_amp, chunk=ec.get("chunk"), num_workers=int(self.tc.get("num_workers", 0)),
            pin_memory=bool(ec.get("pin_memory", False)),
            n_boot=0 if not ec.get("bootstrap_in_training", False) else 1000,
            collate_fn=_collate,
        )
        if self.ctx.is_main and recs:
            for s in recs[:2]:
                log.info("VALID sample [%s] REF: %r | HYP: %r", s["utt_id"], s["ref"], s["hyp"])
        # Free the per-utterance record list immediately; it can be several hundred MB on the full
        # valid set and Python's GC would otherwise keep it alive until the next GC cycle.
        del recs
        wer = summary["wer"]
        reg = " ".join(f"{k}={v['wer']:.4f}" for k, v in summary["by_region"].items())
        host_gib = _host_ram_gib()
        log.info("VALID epoch %d step %d | WER %.4f CER %.4f | host %.1f GiB | %s",
                 self.epoch, self.step, wer, summary["cer"], host_gib, reg)
        acc_fields = {}
        if "align" in summary:
            sa = summary["align"]
            log.info("VALID align | cos %.4f top1 %.4f | qty err %.4f | %d words in %d utts", sa["cos"], sa["top1"],
                     sa["qty_err"], sa["n_words"], sa["n_utts"])
            acc_fields.update(align_cos=sa["cos"], align_top1=sa["top1"], align_qty_err=sa["qty_err"])
        if "accent" in summary:
            ac = summary["accent"]
            log.info("VALID accent | acc %.4f macro-F1 %.4f | recall %s", ac["accuracy"], ac["macro_f1"],
                     " ".join(f"{k}={v:.3f}" for k, v in ac["recall"].items()))
            acc_fields = {"accent_acc": ac["accuracy"], "accent_macro_f1": ac["macro_f1"]}
            w = self.raw.accent.layer_weights()
            if w is not None:
                log.info("accent layer weights (blocks 0..%d): %s", len(w) - 1, " ".join(f"{x:.3f}" for x in w.tolist()))
                acc_fields["accent_layer_weights"] = [round(x, 4) for x in w.tolist()]
        self.metrics.write(kind="valid", step=self.step, epoch=self.epoch, wer=wer, cer=summary["cer"],
                           by_region={k: v["wer"] for k, v in summary["by_region"].items()},
                           host_gib=host_gib, **acc_fields)
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
        _malloc_trim()
        return summary
