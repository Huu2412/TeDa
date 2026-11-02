"""CIF aligner + PhoBERT cache (E02/E03): integration maths, losses, controls, cache, wiring, training."""

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from conftest import tiny_cfg
from teda.data.align_cache import AlignCache
from teda.data.collate import collate
from teda.data.phobert_cache import make_segmenter, unit_vectors, write_cache
from teda.data.tokenizer import BPETokenizer
from teda.data.vimd_dataset import VimdDataset
from teda.models.heads.cif import CIFAligner, cif_integrate
from teda.models.teda_model import build_model
from test_train import make_trainer


# ------------------------------------------------------------------------------------------- fakes for PhoBERT
class FakeTok:
    bos_token_id, eos_token_id, pad_token_id, unk_token_id = 0, 2, 1, 3

    def encode(self, word, add_special_tokens=False):
        return [(ord(c) % 50) + 4 for c in word.replace("_", "")][:: 1] if word else []


class FakeBert(nn.Module):
    """Context-free 'PhoBERT': a fixed embedding per piece id, so chunking cannot change a unit's vector."""

    def __init__(self, dim=16):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = nn.Embedding(64, dim)
        self.emb.weight.data = torch.randn(64, dim, generator=g)

    def forward(self, input_ids, attention_mask):
        return type("O", (), {"last_hidden_state": self.emb(input_ids)})


def _build_cache(tiny, out, unit="syllable", dim=16):
    rows = tiny["rows"]
    seg = make_segmenter(unit)
    units = [seg(r["text_norm"]) for r in rows]
    vecs = unit_vectors(units, FakeTok(), FakeBert(dim), torch.device("cpu"), max_pieces=64, batch_size=8)
    write_cache(out, [r["utt_id"] for r in rows], units, vecs, [r["split"] == "train" for r in rows],
                {"unit": unit, "model": "fake", "text_field": "text_norm"})
    return out


# ------------------------------------------------------------------------------------------- CIF maths
def _loop_cif(h, alpha, n_words):
    """Reference integrate-and-fire written as the classic sequential loop (one utterance)."""
    out, acc, carry_w = [], torch.zeros(h.size(1)), 0.0
    word = torch.zeros(h.size(1))
    mass = 0.0
    for t in range(h.size(0)):
        a = float(alpha[t])
        while a > 1e-12:
            room = 1.0 - mass
            take = min(a, room)
            word = word + take * h[t]
            mass += take
            a -= take
            if mass >= 1.0 - 1e-9 and len(out) < n_words:
                out.append(word.clone())
                word, mass = torch.zeros(h.size(1)), 0.0
    return torch.stack(out)


def test_cif_integrate_matches_the_sequential_loop_and_each_word_collects_unit_mass():
    torch.manual_seed(0)
    T, N = 40, 7
    h = torch.randn(2, T, 6)
    raw = torch.rand(2, T) + 0.1
    alpha = raw * (N / raw.sum(1, keepdim=True))
    e, mass = cif_integrate(h, alpha, N)
    assert torch.allclose(mass, torch.ones(2, N), atol=1e-4)
    for b in range(2):
        assert torch.allclose(e[b], _loop_cif(h[b], alpha[b], N), atol=1e-4)


def test_cif_padding_frames_with_zero_alpha_do_not_change_the_words():
    torch.manual_seed(1)
    h = torch.randn(1, 30, 5)
    a = torch.rand(1, 30) + 0.2
    a = a * (5 / a.sum())
    e1, _ = cif_integrate(h, a, 5)
    h2, a2 = torch.cat([h, torch.randn(1, 10, 5)], 1), torch.cat([a, torch.zeros(1, 10)], 1)
    e2, _ = cif_integrate(h2, a2, 5)
    assert torch.allclose(e1, e2, atol=1e-5)


def test_cif_is_differentiable_in_both_frames_and_alpha():
    h = torch.randn(1, 20, 4, requires_grad=True)
    a = (torch.rand(1, 20) + 0.2)
    a = (a * (4 / a.sum())).requires_grad_(True)
    e, _ = cif_integrate(h, a, 4)
    e.pow(2).sum().backward()
    assert h.grad.abs().sum() > 0 and a.grad.abs().sum() > 0


# ------------------------------------------------------------------------------------------- aligner losses
def _fit_inputs(B=3, T=60, D=12, N=6, dim=10):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(B, T, D, generator=g)
    lens = torch.tensor([T, T - 10, T - 20])[:B]
    n = torch.tensor([N, N - 1, N - 2])[:B]
    tg = torch.randn(B, N, dim, generator=g)
    return h, lens, tg, n


def test_aligner_fits_fixed_targets_and_retrieval_beats_chance():
    torch.manual_seed(0)
    h, lens, tg, n = _fit_inputs()
    al = CIFAligner(12, text_dim=10, proj_hidden=64, dropout=0.0).eval()
    opt = torch.optim.AdamW(al.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        r = al(h, lens, tg, n)
        first = first if first is not None else float(r["loss_align"].detach())
        opt.zero_grad()
        (r["loss_align"] + r["loss_qty"]).backward()
        opt.step()
    assert float(r["loss_align"].detach()) < 0.5 * first, (first, float(r["loss_align"].detach()))
    assert float(r["align_top1"]) > 0.8 and float(r["align_cos"]) > 0.5
    assert float(r["align_words"]) == int(n.sum())


def test_quantity_loss_teaches_alpha_to_sum_to_the_word_count():
    torch.manual_seed(0)
    h, lens, tg, n = _fit_inputs()
    al = CIFAligner(12, text_dim=10, proj_hidden=16, dropout=0.0).eval()
    opt = torch.optim.AdamW(al.parameters(), lr=3e-2)
    e0 = float(al(h, lens, tg, n)["qty_err"])
    for _ in range(200):
        r = al(h, lens, tg, n)
        opt.zero_grad()
        r["loss_qty"].backward()
        opt.step()
    assert float(r["qty_err"]) < 0.3 * e0 or float(r["qty_err"]) < 0.05, (e0, float(r["qty_err"]))


def test_infeasible_or_empty_utterances_are_left_out_without_nan():
    h, lens, tg, n = _fit_inputs()
    al = CIFAligner(12, text_dim=10, proj_hidden=16).eval()
    r = al(h, lens, tg, torch.tensor([6, 0, 500]))        # normal / no words / more words than frames
    assert math.isfinite(float(r["loss_align"])) and float(r["align_utts"]) == 1
    assert float(r["align_words"]) == 6
    r = al(h, lens, tg, torch.zeros(3, dtype=torch.long))  # nothing valid at all: zero loss that still has a graph
    assert float(r["loss_align"]) == 0.0 and r["loss_align"].requires_grad
    r["loss_align"].backward()


def test_shuffle_control_destroys_the_pairing_but_keeps_the_target_statistics():
    torch.manual_seed(0)
    h, lens, tg, n = _fit_inputs()
    real = CIFAligner(12, text_dim=10, target_mode="phobert").eval()
    ctl = CIFAligner(12, text_dim=10, target_mode="shuffle").eval()
    ctl.load_state_dict(real.state_dict())
    wmask = torch.arange(tg.size(1))[None] < n[:, None]
    shuffled = ctl._targets(tg, wmask)
    assert not torch.equal(shuffled[wmask], tg[wmask])
    assert torch.allclose(shuffled[wmask].sort(0).values, tg[wmask].sort(0).values)  # same rows, other positions
    assert torch.equal(real._targets(tg, wmask), tg)
    with pytest.raises(ValueError):
        CIFAligner(12, target_mode="bogus")


# ------------------------------------------------------------------------------------------- cache
def test_cache_roundtrip_standardisation_chunking_and_random_control(tiny_dataset, tmp_path):
    d = _build_cache(tiny_dataset, tmp_path / "cache")
    c = AlignCache(d)
    rows = tiny_dataset["rows"]
    assert len(c.index) == len(rows) and c.dim == 16 and c.meta["unit"] == "syllable"
    words, v = c.get(rows[0]["utt_id"])
    assert words == rows[0]["text_norm"].split() and v.shape == (len(words), 16) and v.dtype == torch.float16
    train = torch.cat([c.get(r["utt_id"])[1] for r in rows if r["split"] == "train"]).float()
    assert train.mean(0).abs().max() < 0.05 and (train.std(0) - 1).abs().max() < 0.1   # standardised on TRAIN words
    # context-free fake model: chunking (max_pieces=64 forces it for long texts) must not change vectors
    seg = make_segmenter("syllable")
    long_units = [seg(" ".join(r["text_norm"] for r in rows[:12]))]
    a = unit_vectors(long_units, FakeTok(), FakeBert(), torch.device("cpu"), max_pieces=24, batch_size=2)[0]
    b = unit_vectors(long_units, FakeTok(), FakeBert(), torch.device("cpu"), max_pieces=4096, batch_size=2)[0]
    assert a.shape == b.shape and np.allclose(a, b, atol=1e-5)
    r1 = AlignCache(d, mode="random_word").get(rows[0]["utt_id"])[1]
    r2 = AlignCache(d, mode="random_word").get(rows[0]["utt_id"])[1]
    assert torch.equal(r1, r2) and not torch.allclose(r1.float(), v.float())     # fixed per word, not PhoBERT
    assert AlignCache(d).n_words("nope") == 0 and "nope" not in c


def test_word_unit_needs_the_segmenter_and_syllable_needs_nothing():
    assert make_segmenter("syllable")("xin chào các bạn") == ["xin", "chào", "các", "bạn"]
    with pytest.raises(ValueError):
        make_segmenter("char")


def test_dataset_and_collate_carry_word_targets(tiny_dataset, tmp_path):
    d = _build_cache(tiny_dataset, tmp_path / "cache")
    tok = BPETokenizer(tiny_dataset["tokenizer"])
    ds = VimdDataset(data_dir=tiny_dataset["root"], manifest="manifest.tsv", split="valid", tokenizer=tok,
                     feat_cfg=tiny_cfg().features, train=False, align_cache=AlignCache(d))
    b = collate([ds[0], ds[1], ds[2]])
    assert b["word_targets"].shape == (3, int(b["n_words"].max()), 16) and b["word_targets"].dtype == torch.float16
    assert b["n_words"].tolist() == [len(ds.rows[i]["text_norm"].split()) for i in range(3)]
    plain = VimdDataset(data_dir=tiny_dataset["root"], manifest="manifest.tsv", split="valid", tokenizer=tok,
                        feat_cfg=tiny_cfg().features, train=False)
    assert "word_targets" not in collate([plain[0]])      # off by default: batches are exactly what E00 sees


# ------------------------------------------------------------------------------------------- model wiring
def _cfg(**align):
    cfg = tiny_cfg()
    cfg.model.align.enabled = True
    cfg.model.align.text_dim = 10
    cfg.model.align.proj_hidden = 16
    for k, v in align.items():
        cfg.model.align[k] = v
    return cfg


def _mbatch(B=3, T=480, U=5, N=4):
    g = torch.Generator().manual_seed(0)
    return {"feats": torch.randn(B, T, 80, generator=g), "feat_lens": torch.tensor([T, T - 40, T - 90][:B]),
            "tokens": torch.randint(2, 30, (B, U), generator=g), "token_lens": torch.tensor([U, U - 1, U][:B]),
            "region_ids": torch.tensor([0, 1, 2][:B]), "word_targets": torch.randn(B, N, 10, generator=g),
            "n_words": torch.tensor([N, N - 1, N][:B])}


def test_align_is_off_by_default_and_validates_its_layer():
    model, _ = build_model(tiny_cfg(), vocab_size=40)
    assert model.align is None
    assert set(model(_mbatch()).keys()) == {"loss", "loss_ctc"}
    with pytest.raises(ValueError):
        build_model(_cfg(layer=99), vocab_size=40)


@pytest.mark.parametrize("layer", ["last", 1])
def test_forward_adds_the_weighted_alignment_terms(layer):
    torch.manual_seed(0)
    model, _ = build_model(_cfg(layer=layer, lambda_align=0.2, lambda_qty=0.1), vocab_size=40)
    model.eval()
    out = model(_mbatch())
    for k in ("loss_align", "loss_qty", "loss_align_term", "align_cos", "align_top1", "qty_err"):
        assert k in out and torch.isfinite(out[k])
    want = out["loss_ctc"] * (0.2 * out["loss_align"] + 0.1 * out["loss_qty"])     # relative: share of the CTC level
    assert float(out["loss_align_term"]) == pytest.approx(float(want), rel=1e-4)
    assert float(out["loss"]) == pytest.approx(float(out["loss_ctc"] + out["loss_align_term"]), rel=1e-4)
    assert model.encoder._captured == []
    model.align_scale = 0.5
    assert float(model(_mbatch())["loss_align_term"]) == pytest.approx(0.5 * float(want), rel=1e-3)


def test_fixed_weighting_does_not_depend_on_the_ctc_level():
    model, _ = build_model(_cfg(weighting="fixed", lambda_align=0.3, lambda_qty=0.0), vocab_size=40)
    model.eval()
    out = model(_mbatch())
    assert float(out["loss_align_term"]) == pytest.approx(0.3 * float(out["loss_align"]), rel=1e-4)


def test_warmup_detach_keeps_alignment_gradients_out_of_the_encoder():
    torch.manual_seed(0)
    model, _ = build_model(_cfg(lambda_align=1.0, lambda_qty=1.0), vocab_size=40)
    model.eval()
    b = _mbatch()
    p = next(q for q in model.encoder.parameters() if q.requires_grad)

    def grad_of(loss):
        return torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0]

    model.align_detached = True
    out = model(b)
    g_detached = grad_of(out["loss"])
    enc, lens = model.encoder(b["feats"], b["feat_lens"])
    g_ctc = grad_of(model.ctc.loss(enc, lens, b["tokens"], b["token_lens"]))
    assert torch.allclose(g_detached, g_ctc, atol=1e-5, rtol=1e-4), "alignment leaked into the encoder during warm-up"
    assert any(q.grad is None for q in model.align.parameters())     # (no backward yet) just sanity: they exist
    model.align_detached = False
    g_attached = grad_of(model(b)["loss"])
    assert not torch.allclose(g_attached, g_ctc, atol=1e-5, rtol=1e-4), "alignment never reaches the encoder after warm-up"
    out = model(b)
    out["loss"].backward()
    assert all(q.grad is not None for q in model.align.parameters())


def test_encode_eval_reports_alignment_and_accent_together():
    cfg = _cfg()
    cfg.model.accent.enabled = True
    model, _ = build_model(cfg, vocab_size=40)
    b = _mbatch()
    r = model.encode_eval(b["feats"], b["feat_lens"], word_targets=b["word_targets"], n_words=b["n_words"])
    assert {"logits", "lens", "accent_logits", "align"} <= set(r) and r["accent_logits"].shape == (3, 3)
    r2 = model.encode_eval(b["feats"], b["feat_lens"])
    assert "align" not in r2            # no cache in the dataset (e.g. evaluate.py on test): CTC decoding still works


# ------------------------------------------------------------------------------------------- training end to end
def test_trainer_with_alignment_logs_validates_and_resumes(tiny_dataset, tmp_path):
    cache = _build_cache(tiny_dataset, tmp_path / "cache")
    out = tmp_path / "run"
    tr, model = make_trainer(tiny_dataset, out, align_cache=cache, epochs=2)
    res = tr.fit()
    assert res["status"] == "done"
    rows = [json.loads(line) for line in open(out / "metrics.jsonl", encoding="utf-8")]
    train_rows = [r for r in rows if r["kind"] == "train"]
    assert train_rows and all(k in train_rows[0] for k in ("loss_align", "loss_qty", "align_cos", "align_top1"))
    valid = [r for r in rows if r["kind"] == "valid"]
    assert valid and all(math.isfinite(r["align_cos"]) and 0 <= r["align_top1"] <= 1 and r["align_qty_err"] >= 0 for r in valid)
    summary = tr.validate()
    assert summary["align"]["n_words"] == sum(len(r["text_norm"].split()) for r in tiny_dataset["rows"] if r["split"] == "valid")
    again, _ = make_trainer(tiny_dataset, out, align_cache=cache, epochs=2)
    assert again.fit()["status"] == "done"


def test_trainer_with_accent_and_controls_runs(tiny_dataset, tmp_path):
    cache = _build_cache(tiny_dataset, tmp_path / "cache")
    for mode in ("random_word", "shuffle"):
        tr, _ = make_trainer(tiny_dataset, tmp_path / f"run_{mode}", accent=True, align_cache=cache,
                             align={"target_mode": mode}, epochs=1)
        assert tr.fit()["status"] == "done"
