"""Accent head (E01): block-output capture, masking, class weights, training / evaluation wiring."""

import json
import math

import numpy as np
import pytest
import torch

from conftest import tiny_cfg
from teda.engine.evaluator import accent_metrics
from teda.models.heads.accent_head import AccentHead
from teda.models.teda_model import build_model
from test_train import _run, _tiny_exp_file, make_trainer


def _cfg(**accent):
    cfg = tiny_cfg()
    cfg.model.accent.enabled = True
    for k, v in accent.items():
        cfg.model.accent[k] = v
    return cfg


def _batch(B=3, T=480, U=5):
    return {"feats": torch.randn(B, T, 80), "feat_lens": torch.tensor([T, T - 40, T - 90][:B]),
            "tokens": torch.randint(2, 30, (B, U)), "token_lens": torch.tensor([U, U - 1, U][:B]),
            "region_ids": torch.tensor([0, 1, 2][:B])}


def test_accent_head_is_off_by_default_and_costs_nothing():
    model, _ = build_model(tiny_cfg(), vocab_size=40)
    assert model.accent is None and not model.encoder._hooks
    out = model(_batch())
    assert set(out) == {"loss", "loss_ctc"}


def test_every_block_output_is_captured_once_per_forward():
    model, _ = build_model(_cfg(), vocab_size=40)
    n_blocks = len(model.encoder.net.encoders)
    model.train()
    for _ in range(2):  # captured outputs must not pile up across calls
        model(_batch())
        assert model.encoder._captured == [], "outputs were not consumed by the head"
    enc, lens = model.encoder(_batch()["feats"], _batch()["feat_lens"])
    outs = model.encoder.pop_layer_outputs()
    assert len(outs) == n_blocks and all(o.shape == enc.shape for o in outs)
    model.eval()
    model.encode_logits(_batch()["feats"], _batch()["feat_lens"])
    assert model.encoder._captured == []  # eval without the accent task releases them too


def test_weighted_mix_trains_and_gradients_reach_the_mixing_weights_and_the_encoder():
    torch.manual_seed(0)
    model, _ = build_model(_cfg(weighting="fixed", **{"lambda": 1.0}), vocab_size=40)
    w0 = model.accent.layer_weights().clone()
    assert torch.allclose(w0, torch.full_like(w0, 1 / len(w0)))  # starts uniform
    # (1) only the head is optimised, on one fixed batch, with no dropout: its CE must fall clearly
    opt = torch.optim.AdamW(model.accent.parameters(), lr=2e-2)
    model.eval()
    b = _batch()
    first = None
    for _ in range(80):
        out = model(b)
        assert torch.isfinite(out["loss"]) and "loss_accent" in out and "accent_acc" in out
        out["loss"].backward()
        assert model.accent.layer_logits.grad is not None
        opt.step()
        opt.zero_grad()
        first = first if first is not None else float(out["loss_accent"])
    assert float(out["loss_accent"]) < 0.7 * first, (first, float(out["loss_accent"]))
    assert not torch.allclose(model.accent.layer_weights(), w0), "mixing weights never moved"
    # (2) the accent loss also sends gradient into the encoder blocks (that is what shapes the representation)
    model.zero_grad()
    out = model(b)
    out["_graph_acc"].backward()
    trainable_blocks = [blk for blk in model.encoder.net.encoders if any(q.requires_grad for q in blk.parameters())]
    assert trainable_blocks, "the tiny config should leave some blocks trainable"
    for blk in trainable_blocks:
        assert any(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in blk.parameters() if q.requires_grad)


def test_aux_scale_zero_removes_the_accent_gradient_but_keeps_ctc():
    model, _ = build_model(_cfg(), vocab_size=40)
    model.train()
    model.aux_scale = 0.0
    out = model(_batch())
    assert float(out["loss"].detach()) == pytest.approx(float(out["loss_ctc"]))
    out["loss"].backward()
    g = model.accent.layer_logits.grad
    assert g is None or float(g.abs().sum()) == 0.0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.ctc.parameters())


def test_padding_frames_do_not_change_the_pooled_prediction():
    torch.manual_seed(0)
    head = AccentHead(dim=16, num_layers=3, pooling="asp").eval()
    outs = [torch.randn(2, 10, 16) for _ in range(3)]
    last = torch.randn(2, 10, 16)
    lens = torch.tensor([10, 6])
    a = head(outs, last, lens)
    noisy = [o.clone() for o in outs]
    for o in noisy:
        o[1, 6:] = torch.randn(4, 16) * 100  # garbage in the padded tail of utterance 1
    b = head(noisy, last, lens)
    assert torch.allclose(a, b, atol=1e-5)


def test_class_weights_are_inverse_frequency_with_mean_one_and_checkpointed():
    head = AccentHead(dim=8, num_layers=2)
    head.set_class_weights([700, 100, 200])
    w = head.class_weight
    assert float(w.mean()) == pytest.approx(1.0) and w[1] > w[2] > w[0]
    assert "class_weight" in head.state_dict()


def test_layers_last_has_no_mixing_parameters_and_bad_options_are_rejected():
    model, _ = build_model(_cfg(layers="last"), vocab_size=40)
    assert model.accent.layer_weights() is None and not hasattr(model.accent, "layer_logits")
    model.train()
    assert torch.isfinite(model(_batch())["loss"])
    with pytest.raises(ValueError):
        AccentHead(8, 2, layers="middle")
    with pytest.raises(ValueError):
        AccentHead(8, 2, pooling="max")


def test_accent_metrics_values():
    m = accent_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["recall"]["North"] == pytest.approx(0.5) and m["recall"]["Central"] == pytest.approx(1.0)
    assert np.array(m["confusion"]["matrix"]).sum() == 6
    f1 = [2 * 0.5 * 0.5 / 1.0, 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0), 2 * 1.0 * 0.5 / 1.5]
    assert m["macro_f1"] == pytest.approx(sum(f1) / 3)


def test_trainer_with_accent_logs_metrics_and_validation_reports_accent(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    tr, model = make_trainer(tiny_dataset, out, accent=True, epochs=2)
    res = tr.fit()
    assert res["status"] == "done"
    rows = [json.loads(line) for line in open(out / "metrics.jsonl", encoding="utf-8")]
    train_rows = [r for r in rows if r["kind"] == "train"]
    assert train_rows and all("loss_accent" in r and 0.0 <= r["accent_acc"] <= 1.0 for r in train_rows)
    valid = [r for r in rows if r["kind"] == "valid"]
    assert valid and all(math.isfinite(r["accent_acc"]) and len(r["accent_layer_weights"]) == 3 for r in valid)
    assert abs(sum(valid[-1]["accent_layer_weights"]) - 1.0) < 1e-3
    assert "accent" in tr.validate() and tr.validate()["accent"]["confusion"]["labels"] == ["North", "Central", "South"]
    # the head is part of last.pt, so an interrupted E01 run resumes with its accent weights
    again, _ = make_trainer(tiny_dataset, out, accent=True, epochs=2)
    assert again.fit()["status"] == "done"


def test_two_process_ddp_with_accent_then_evaluate_script_reports_accent(tiny_dataset, tmp_path):
    """DDP + forward hooks + class weights from train.py + evaluate.py output, as on Kaggle (gloo instead of nccl)."""
    import sys

    import yaml

    if sys.platform == "win32":
        pytest.skip("torch.distributed.run multi-process DDP requires libuv which is not built in Windows PyTorch")

    exp = _tiny_exp_file(tiny_dataset, tmp_path)
    cfg = yaml.safe_load(exp.read_text(encoding="utf-8"))
    cfg.setdefault("model", {})["accent"] = {"enabled": True, "ramp_steps": 2, "start_step": 0}
    exp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = _run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "scripts/train.py", "--exp", str(exp)])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    out = tmp_path / "out" / "tiny_exp"
    log = (out / "train.log").read_text(encoding="utf-8")
    assert "accent class counts (train)" in log and "VALID accent" in log and "accent layer weights" in log
    r = _run([sys.executable, "scripts/evaluate.py", "--exp", str(exp), "--split", "valid"])
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    res = json.loads((out / "results_valid.json").read_text(encoding="utf-8"))
    assert set(res["accent"]) == {"accuracy", "macro_f1", "recall", "confusion"} and res["n_utts"] == 6
    assert sum(map(sum, res["accent"]["confusion"]["matrix"])) == 6


def test_relative_weighting_makes_lambda_a_share_of_ctc_regardless_of_its_scale():
    torch.manual_seed(0)
    model, _ = build_model(_cfg(weighting="relative", ramp_steps=0, **{"lambda": 0.1}), vocab_size=40)
    model.train()
    out = model(_batch())
    ce = float(out["loss_accent"])
    # accent term = lambda * CTC * CE / ln(3)
    expected = 0.1 * float(out["loss_ctc"]) * ce / math.log(3)
    assert float(out["loss_accent_term"]) == pytest.approx(expected, rel=1e-4)
    # the CTC factor is detached: backward works and gives finite gradients everywhere
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_fixed_weighting_is_the_plain_sum_and_bad_value_is_rejected():
    model, _ = build_model(_cfg(weighting="fixed", **{"lambda": 0.3}), vocab_size=40)
    model.train()
    out = model(_batch())
    assert float(out["loss"].detach()) == pytest.approx(float(out["loss_ctc"]) + 0.3 * float(out["loss_accent"]), rel=1e-4)
    with pytest.raises(ValueError):
        build_model(_cfg(weighting="auto"), vocab_size=40)


def test_relative_weighting_gives_a_larger_accent_gradient_than_fixed_when_ctc_is_large():
    """The point of the change: with CTC in the tens-hundreds, fixed lambda leaves the accent share tiny."""
    def accent_share(weighting, lam):
        torch.manual_seed(0)
        model, _ = build_model(_cfg(weighting=weighting, ramp_steps=0, **{"lambda": lam}), vocab_size=40)
        model.train()
        torch.manual_seed(1)
        out = model(_batch())
        params = [p for p in model.encoder.net.encoders[-1].parameters() if p.requires_grad]
        ga = torch.autograd.grad(out["_graph_acc"], params, retain_graph=True, allow_unused=True)
        gc = torch.autograd.grad(out["_graph_ctc"], params, retain_graph=True, allow_unused=True)
        n = lambda gs: float(torch.cat([g.flatten() for g in gs if g is not None]).norm())
        return n(ga) / (n(ga) + n(gc)), float(out["loss_ctc"])

    fixed, ctc = accent_share("fixed", 0.2)
    rel, _ = accent_share("relative", 0.2)
    assert ctc > 5 and rel > 2 * fixed, (fixed, rel, ctc)


def test_trainer_grad_probe_logs_on_rank0_and_does_not_change_training(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    tr, _ = make_trainer(tiny_dataset, out, accent=True, epochs=1)
    tr.probe_every = 1
    tr.fit()
    rows = [json.loads(line) for line in open(out / "metrics.jsonl", encoding="utf-8")]
    probes = [r for r in rows if r["kind"] == "grad_probe"]
    assert probes and all(0.0 <= r["acc_share"] <= 1.0 and -1.0001 <= r["cos"] <= 1.0001 for r in probes)


def test_warmup_phase_trains_the_head_but_sends_no_gradient_into_the_encoder():
    torch.manual_seed(0)
    model, _ = build_model(_cfg(), vocab_size=40)
    model.train()
    b = _batch()
    model.accent_detached = True
    out = model(b)
    assert float(out["loss_accent_term"]) == pytest.approx(float(out["loss_accent"]), rel=1e-5)  # plain CE for the head
    params = [p for p in model.encoder.net.encoders[-1].parameters() if p.requires_grad]
    g = torch.autograd.grad(out["_graph_acc"], params, retain_graph=True, allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in g), "accent leaked gradient into the encoder"
    out["loss"].backward()
    assert model.accent.layer_logits.grad is not None and any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.accent.mlp.parameters())
    # the CTC head still learns as usual
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.ctc.parameters())

    model.zero_grad()
    model.accent_detached = False
    out = model(b)
    g = torch.autograd.grad(out["_graph_acc"], params, retain_graph=True, allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in g), "after start_step the encoder must receive accent gradient"


def test_trainer_switches_from_detached_to_attached_at_start_step(tiny_dataset, tmp_path):
    tr, model = make_trainer(tiny_dataset, tmp_path / "run", accent=True, epochs=2)
    tr.acc_start, tr.acc_ramp, tr.probe_every = 3, 2, 1
    tr.fit()
    rows = [json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl", encoding="utf-8")]
    probes = {r["step"]: r for r in rows if r["kind"] == "grad_probe"}
    early = [r for s, r in probes.items() if s <= 3]
    late = [r for s, r in probes.items() if s > 4]
    assert early and late
    assert all(r["grad_acc"] == 0.0 for r in early), "no accent gradient before start_step"
    assert any(r["grad_acc"] > 0.0 for r in late), "accent gradient must reach the encoder after start_step"
