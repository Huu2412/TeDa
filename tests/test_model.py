import pytest
import torch

from conftest import TINY_ENCODER, tiny_cfg
from teda.data.features import subsampled_length
from teda.engine.decoding import ctc_greedy
from teda.models.encoder import SpeechEncoder, build_chunkformer, load_pretrained
from teda.models.heads.ctc_head import CTCHead
from teda.models.teda_model import build_model
from teda.utils.freeze import apply_freeze


def _batch(B=2, T=480, U=6):
    return {"feats": torch.randn(B, T, 80), "feat_lens": torch.tensor([T, T - 40][:B]),
            "tokens": torch.randint(2, 30, (B, U)), "token_lens": torch.tensor([U, U - 1][:B])}


def test_subsampled_length_matches_real_encoder():
    enc = SpeechEncoder(build_chunkformer(TINY_ENCODER)).eval()
    for T in [60, 99, 100, 400, 401, 1003, 2999]:
        with torch.no_grad():
            out, lens = enc(torch.randn(1, T, 80), torch.tensor([T]))
        assert int(lens[0]) == out.size(1) == subsampled_length(T), T


def test_forward_backward_and_freeze():
    model, rep = build_model(tiny_cfg(), vocab_size=40)
    frozen_params = [p for p in model.encoder.net.embed.parameters()] + list(model.encoder.net.encoders[0].parameters())
    assert all(not p.requires_grad for p in frozen_params)
    assert rep["model_total"][1] == sum(p.numel() for p in frozen_params)

    before_f = [p.detach().clone() for p in frozen_params]
    trainable = [p for p in model.parameters() if p.requires_grad]
    before_t = [p.detach().clone() for p in trainable]
    opt = torch.optim.AdamW(trainable, lr=1e-2)

    model.train()
    for _ in range(3):  # several steps => several random dynamic-chunk draws
        loss = model(_batch())["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert all(torch.equal(a, b) for a, b in zip(before_f, frozen_params)), "frozen weights changed"
    assert any(not torch.equal(a, b) for a, b in zip(before_t, trainable)), "trainable weights did not change"


def test_frozen_blocks_stay_in_eval_but_encoder_keeps_training_mode():
    model, _ = build_model(tiny_cfg(), vocab_size=40)
    model.train()
    net = model.encoder.net
    assert net.training, "encoder must stay in train mode so dynamic chunk selection keeps working"
    assert not net.embed.training and not net.encoders[0].training
    assert net.encoders[1].training and model.ctc.training
    model.eval()
    model.train()
    assert not net.encoders[0].training  # still frozen-eval after toggling


def test_freeze_index_validation():
    net = build_chunkformer(TINY_ENCODER)
    with pytest.raises(ValueError):
        apply_freeze(net, freeze_layers=[0, 3])  # only blocks 0..2 exist


def test_eval_full_and_chunked_give_valid_outputs():
    model, _ = build_model(tiny_cfg(), vocab_size=40)
    model.eval()
    b = _batch(T=800)
    full, lens = model.encode_logits(b["feats"], b["feat_lens"])
    chunked, lens_c = model.encode_logits(b["feats"], b["feat_lens"], chunk=(16, 16, 16))
    assert full.shape == chunked.shape and torch.equal(lens, lens_c)
    assert torch.isfinite(full).all() and torch.isfinite(chunked).all()
    assert not torch.allclose(full, chunked), "chunked decoding should actually restrict the context"


def test_load_pretrained_prefix_handling_and_refusal(tmp_path):
    src = build_chunkformer(TINY_ENCODER)
    sd = {"encoder." + k: v for k, v in src.state_dict().items()}  # exported bundles keep the 'encoder.' prefix
    sd["quantizer.codebook"] = torch.zeros(3)                       # pretraining-only tensor must be ignored
    torch.save(sd, tmp_path / "pytorch_model.pt")

    dst = build_chunkformer(TINY_ENCODER)
    rep = load_pretrained(dst, tmp_path / "pytorch_model.pt")
    assert rep["prefix_stripped"] == "encoder." and rep["coverage"] == 1.0 and rep["unexpected"]
    assert all(torch.equal(a, b) for a, b in zip(src.state_dict().values(), dst.state_dict().values()))

    other = build_chunkformer({**TINY_ENCODER, "output_size": 64, "linear_units": 128})
    with pytest.raises(RuntimeError, match="matched"):
        load_pretrained(other, tmp_path / "pytorch_model.pt")  # wrong architecture must not pass silently


def test_ctc_loss_matches_reference_and_handles_infeasible():
    torch.manual_seed(0)
    head = CTCHead(8, 10)
    h = torch.randn(2, 12, 8)
    lens = torch.tensor([12, 9])
    tokens = torch.tensor([[3, 4, 4, 5], [6, 7, 0, 0]])
    tl = torch.tensor([4, 2])
    got = head.loss(h, lens, tokens, tl)
    logp = torch.log_softmax(head.logits(h), -1).transpose(0, 1)
    ref = torch.nn.functional.ctc_loss(logp, torch.tensor([3, 4, 4, 5, 6, 7]), lens, tl, blank=0, reduction="sum") / 2
    assert torch.allclose(got, ref, atol=1e-5)
    # 3 frames cannot emit 5 tokens: zero_infinity keeps the loss finite instead of inf
    bad = head.loss(h[:1, :3], torch.tensor([3]), torch.tensor([[2, 3, 4, 5, 6]]), torch.tensor([5]))
    assert torch.isfinite(bad)


def test_ctc_greedy_collapses_repeats_and_blanks():
    V = 5
    path = [1, 1, 0, 1, 2, 2, 0, 0, 3]
    logits = torch.full((1, len(path), V), -5.0)
    for t, k in enumerate(path):
        logits[0, t, k] = 5.0
    assert ctc_greedy(logits, torch.tensor([len(path)])) == [[1, 1, 2, 3]]
    assert ctc_greedy(logits, torch.tensor([3])) == [[1]]  # respects the length


def test_accent_head_forward_and_loss():
    from teda.models.heads.accent_head import AccentHead

    head = AccentHead(dim=16, num_classes=3, dropout=0.0)
    enc = torch.randn(4, 20, 16)
    lens = torch.tensor([20, 15, 10, 5])
    labels = torch.tensor([0, 1, 2, 0])

    logits = head.logits(enc, lens)
    assert logits.shape == (4, 3)

    loss, acc = head.loss_and_acc(enc, lens, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0


def test_teda_model_with_accent_head():
    cfg = tiny_cfg()
    cfg.model = {"ctc_dropout": 0.1, "accent": {"enabled": True, "weight": 0.5, "num_classes": 3}}
    model, rep = build_model(cfg, vocab_size=40)
    assert model.accent is not None and model.weight_accent == 0.5

    batch = _batch()
    batch["region_ids"] = torch.tensor([0, 2])

    out = model(batch)
    assert "loss" in out and "loss_ctc" in out and "loss_accent" in out and "accent_acc" in out
    assert torch.isclose(out["loss"], out["loss_ctc"] + 0.5 * out["loss_accent"])

    out["loss"].backward()
    trainable_accent = [p for p in model.accent.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in trainable_accent)

