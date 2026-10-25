"""Training-loop tests on CPU with a tiny random encoder and synthetic audio.

They cover what can go wrong on Kaggle without a GPU: frozen weights staying frozen, checkpoint /
resume, accumulation boundaries, the 2-process DDP path (gloo backend instead of nccl), evaluate.py
and sweep.py. fp16 autocast / GradScaler need a CUDA device and are NOT exercised here.
"""

import json
import math
import os
import subprocess
import sys

import pytest
import torch
import yaml

from conftest import ROOT, TINY_ENCODER, tiny_cfg
from teda.data.tokenizer import BPETokenizer
from teda.data.vimd_dataset import VimdDataset
from teda.engine.trainer import Trainer, make_lr_lambda
from teda.models.teda_model import build_model
from teda.utils import dist as D
from teda.utils.ckpt import load_ckpt, read_json


def make_trainer(tiny, out, **train_over):
    cfg = tiny_cfg(**train_over)
    cfg.data.data_dir = str(tiny["root"])
    tok = BPETokenizer(tiny["tokenizer"])
    common = dict(data_dir=tiny["root"], manifest="manifest.tsv", tokenizer=tok, feat_cfg=cfg.features)
    train_ds = VimdDataset(split="train", train=True, drop_infeasible_ctc=True, **common)
    valid_ds = VimdDataset(split="valid", train=False, **common)
    torch.manual_seed(0)
    model, _ = build_model(cfg, tok.vocab_size)
    return Trainer(cfg, model, train_ds, valid_ds, tok, D.DistContext(), out), model


def test_lr_schedule_warmup_then_cosine_to_the_floor():
    f = make_lr_lambda(warmup=10, total=110, min_ratio=0.05)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(60) == pytest.approx(0.525)
    assert f(110) == pytest.approx(0.05) and f(10_000) == pytest.approx(0.05)


def test_one_batch_can_be_overfitted():
    """If the loss cannot fall on a single fixed batch, the model / CTC head wiring is broken."""
    torch.manual_seed(0)
    model, _ = build_model(tiny_cfg(), vocab_size=40)
    batch = {"feats": torch.randn(3, 400, 80), "feat_lens": torch.tensor([400, 380, 360]),
             "tokens": torch.randint(2, 30, (3, 5)), "token_lens": torch.tensor([5, 4, 5])}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    losses = []
    for _ in range(60):
        loss = model(batch)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert min(losses[-10:]) < 0.5 * losses[0], (losses[0], losses[-10:])


def test_fit_writes_artifacts_keeps_frozen_weights_and_is_idempotent(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    tr, model = make_trainer(tiny_dataset, out)
    frozen0 = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen0
    res = tr.fit()
    assert res["status"] == "done" and res["step"] == tr.step > 0
    for name in ("last.pt", "best.pt", "done.json", "metrics.jsonl"):
        assert (out / name).exists(), name
    for n, p in model.named_parameters():
        if n in frozen0:
            assert torch.equal(p, frozen0[n]), f"frozen parameter {n} changed during training"
    rows = [json.loads(line) for line in open(out / "metrics.jsonl", encoding="utf-8")]
    assert any(r["kind"] == "valid" and math.isfinite(r["wer"]) for r in rows)
    assert all(math.isfinite(r["loss_ctc"]) for r in rows if r["kind"] == "train")

    best = load_ckpt(out / "best.pt")
    assert best["valid_wer"] == res["best_valid_wer"] and "cfg" in best and "model" in best
    again, _ = make_trainer(tiny_dataset, out)
    assert again.fit() == read_json(out / "done.json")  # a finished run is a no-op, not a retrain


def test_interrupted_run_resumes_mid_epoch_and_finishes_with_the_same_step_count(tiny_dataset, tmp_path):
    clean, _ = make_trainer(tiny_dataset, tmp_path / "clean", epochs=2)
    total = clean.fit()["step"]

    out = tmp_path / "interrupted"
    tr, _ = make_trainer(tiny_dataset, out, epochs=2, time_check_every=2, max_hours=1e-9)
    res = tr.fit()
    assert res["status"] == "time_limit" and res["step"] == 2
    st = load_ckpt(out / "last.pt")
    assert st["epoch"] == 0 and st["micro_in_epoch"] == 2 and st["step"] == 2  # stopped inside epoch 0
    assert not (out / "done.json").exists()

    tr2, _ = make_trainer(tiny_dataset, out, epochs=2)
    res2 = tr2.fit()
    assert res2["status"] == "done" and res2["step"] == total  # skipped the 2 finished batches, ran the rest


def test_gradient_accumulation_uses_the_exact_length_of_every_epoch(tiny_dataset, tmp_path):
    tr, _ = make_trainer(tiny_dataset, tmp_path / "run", accum_grad=2, epochs=3)
    tr.fit()
    expected = 0
    for e in range(3):
        tr.sampler.set_epoch(e)
        expected += math.ceil(len(tr.sampler) / 2)
    assert tr.step == expected


def test_resume_with_changed_freeze_settings_is_refused_with_a_clear_message(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    tr, _ = make_trainer(tiny_dataset, out, epochs=1, time_check_every=1, max_hours=1e-9)
    tr.fit()  # leaves last.pt
    cfg_over = tiny_cfg(epochs=1)
    cfg_over.encoder.freeze_layers = [0, 1]  # different set of trainable parameters than the checkpoint
    cfg_over.data.data_dir = str(tiny_dataset["root"])
    tok = BPETokenizer(tiny_dataset["tokenizer"])
    common = dict(data_dir=tiny_dataset["root"], manifest="manifest.tsv", tokenizer=tok, feat_cfg=cfg_over.features)
    model, _ = build_model(cfg_over, tok.vocab_size)
    tr2 = Trainer(cfg_over, model, VimdDataset(split="train", train=True, **common),
                  VimdDataset(split="valid", train=False, **common), tok, D.DistContext(), out)
    with pytest.raises(RuntimeError, match="freeze"):
        tr2.fit()


# ------------------------------------------------------------------ scripts, 2-process DDP
def _tiny_exp_file(tiny, tmp_path, **train_over):
    exp = {
        "exp_name": "tiny_exp",
        "data": {"data_dir": str(tiny["root"]), "tokenizer_model": str(tiny["tokenizer"])},
        "encoder": {"encoder_conf": TINY_ENCODER, "input_dim": 80, "overrides": {}, "freeze_embed": True, "freeze_layers": [0]},
        "train": {"epochs": 2, "max_frames_per_batch": 1200, "accum_grad": 1, "num_workers": 0, "warmup_steps": 2,
                  "log_every": 1, "save_every_steps": 1000, "max_hours": None, "lr_encoder": 1e-3, "lr_head": 1e-2,
                  **train_over},
        "eval": {"batch_frames": 3000},
        "paths": {"out_root": str(tmp_path / "out")},
    }
    path = tmp_path / "tiny_exp.yaml"
    path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    return path


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["USE_LIBUV"] = "0"
    return env


def _run(cmd, timeout=900):
    return subprocess.run(cmd, cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=timeout)


def test_two_process_ddp_train_then_evaluate_with_test_guard(tiny_dataset, tmp_path):
    if sys.platform == "win32":
        pytest.skip("torch.distributed.run multi-process DDP requires libuv which is not built in Windows PyTorch")
    exp = _tiny_exp_file(tiny_dataset, tmp_path)
    r = _run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "scripts/train.py", "--exp", str(exp)])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    out = tmp_path / "out" / "tiny_exp"
    done = read_json(out / "done.json")
    assert done["status"] == "done" and done["step"] > 0 and (out / "best.pt").exists()
    assert "world_size=2" in (out / "train.log").read_text(encoding="utf-8")

    r = _run([sys.executable, "scripts/evaluate.py", "--exp", str(exp), "--split", "valid"])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    res = json.loads((out / "results_valid.json").read_text(encoding="utf-8"))
    assert res["n_utts"] == 6 and set(res["by_region"]) == {"North", "Central", "South"} and "wer_ci95" in res
    assert (out / "hyps_valid.tsv").exists() and (out / "errors_valid.json").exists()

    r = _run([sys.executable, "scripts/evaluate.py", "--exp", str(exp), "--split", "test"])
    assert r.returncode != 0 and "--final" in (r.stdout + r.stderr)  # the test split is protected
    assert not (out / "results_test.json").exists()
    r = _run([sys.executable, "scripts/evaluate.py", "--exp", str(exp), "--split", "test", "--final"])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert json.loads((out / "results_test.json").read_text(encoding="utf-8"))["final"] is True


def test_sweep_runs_each_config_once_and_writes_the_summary(tiny_dataset, tmp_path):
    exp = _tiny_exp_file(tiny_dataset, tmp_path)
    sweep = {
        "base_exp": str(exp), "out_dir": str(tmp_path / "sweep"), "common_set": {"train.epochs": 1},
        "runs": [{"name": "free", "set": {"encoder.freeze_embed": False, "encoder.freeze_layers": []}},
                 {"name": "low2", "set": {"encoder.freeze_embed": True, "encoder.freeze_layers": [0, 1]}}],
    }
    sw = tmp_path / "sweep.yaml"
    sw.write_text(yaml.safe_dump(sweep), encoding="utf-8")
    r = _run([sys.executable, "scripts/sweep.py", "--sweep", str(sw), "--nproc", "1", "--set", "train.log_every=2"])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    for n in ("free", "low2"):  # --set reaches every run (and the sweep's own settings are kept)
        run_cfg = yaml.safe_load((tmp_path / "sweep" / n / "config.yaml").read_text(encoding="utf-8"))
        assert run_cfg["train"]["log_every"] == 2 and run_cfg["train"]["epochs"] == 1
    rows = (tmp_path / "sweep" / "summary.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3 and rows[0].startswith("name\tvalid_wer")
    assert {line.split("\t")[0] for line in rows[1:]} == {"free", "low2"}
    # trainable-parameter counts must differ between the two freeze settings
    logs = [(tmp_path / "sweep" / n / "train.log").read_text(encoding="utf-8") for n in ("free", "low2")]
    assert logs[0] != logs[1]
    # a second invocation finds both runs done and does no training
    r2 = _run([sys.executable, "scripts/sweep.py", "--sweep", str(sw), "--nproc", "1"])
    assert r2.returncode == 0 and "===" not in r2.stdout


def test_trainer_with_accent_head(tiny_dataset, tmp_path):
    out = tmp_path / "accent_run"
    tr, model = make_trainer(tiny_dataset, out, epochs=1, log_every=1)
    from teda.models.heads.accent_head import AccentHead
    model.accent = AccentHead(model.encoder.output_size, num_classes=3)
    model.weight_accent = 0.5
    tr.opt = tr._build_optimizer()
    res = tr.fit()
    assert res["status"] == "done" and res["step"] > 0
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    train_metrics = [m for m in metrics if m.get("kind") == "train"]
    assert any("loss_accent" in m for m in train_metrics)

