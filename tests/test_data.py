import random

import pytest
import torch

from teda.data import features as F
from teda.data.collate import collate
from teda.data.groups import utt_groups
from teda.data.sampler import DynamicBatchSampler
from teda.data.tokenizer import BLANK_ID, UNK_ID, BPETokenizer
from teda.data.vimd_dataset import VimdDataset, ctc_required_frames

FEAT_CFG = {"num_mel_bins": 80, "dither_train": 0.0, "dither_eval": 0.0, "spec_aug": {"enabled": False}}


# ------------------------------------------------------------------------------ sampler
def _lengths(n=500, seed=1):
    rng = random.Random(seed)
    return [rng.randint(100, 900) for _ in range(n)]


def test_sampler_covers_every_utterance_once_and_respects_frame_budget():
    lengths = _lengths()
    seen = []
    for rank in range(2):
        s = DynamicBatchSampler(lengths, 3000, 2, rank, shuffle=True, seed=3, pad_to_world=False)
        s.set_epoch(4)
        for b in s:
            assert max(lengths[i] for i in b) * len(b) <= 3000  # padded size, not just the sum
            seen += b
    assert sorted(seen) == list(range(len(lengths)))


def test_sampler_gives_every_rank_the_same_number_of_batches_when_padded():
    lengths = _lengths(n=333)
    counts = []
    for rank in range(3):
        s = DynamicBatchSampler(lengths, 2500, 3, rank, seed=0)
        s.set_epoch(1)
        assert len(list(s)) == len(s)
        counts.append(len(s))
    assert len(set(counts)) == 1, "unequal step counts across ranks would hang DDP"


def test_sampler_is_deterministic_and_changes_with_epoch():
    lengths = _lengths()

    def run(seed, epoch):
        s = DynamicBatchSampler(lengths, 3000, seed=seed)
        s.set_epoch(epoch)
        return list(s)

    assert run(5, 2) == run(5, 2)
    assert run(5, 2) != run(5, 3)
    assert run(5, 2) != run(6, 2)


def test_sampler_skip_resumes_in_the_middle_of_an_epoch():
    s = DynamicBatchSampler(_lengths(), 3000, seed=1)
    s.set_epoch(2)
    full = list(s)
    s.set_epoch(2, skip=3)
    assert list(s) == full[3:]
    assert len(s) == len(full)  # __len__ is the whole epoch, independent of skip


def test_sampler_max_batch_size_and_too_long_utterance():
    s = DynamicBatchSampler([100] * 40, 100_000, shuffle=False, max_batch_size=4)
    assert all(len(b) <= 4 for b in s)
    with pytest.raises(ValueError, match="longer than"):
        DynamicBatchSampler([100, 5000], 1000)


# ---------------------------------------------------------------------------- tokenizer
def test_tokenizer_layout_and_roundtrip(tiny_dataset):
    tok = BPETokenizer(tiny_dataset["tokenizer"])
    assert tok.sp.pad_id() == BLANK_ID == 0 and tok.sp.unk_id() == UNK_ID == 1
    assert tok.vocab_size <= 40
    text = "xin chào các bạn"
    ids = tok.encode(text)
    assert BLANK_ID not in ids and all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text
    assert tok.decode([BLANK_ID] + ids + [BLANK_ID]) == text  # blanks are dropped when decoding
    # unseen characters map to <unk>; SentencePiece merges a run of them into ONE <unk> token
    assert UNK_ID in tok.encode("qqq") and 0 < tok.unk_rate(["qqq"]) < 1 and tok.unk_rate(["xin chào"]) == 0


def test_tokenizer_missing_model_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_tokenizer"):
        BPETokenizer(tmp_path / "nope.model")


# ------------------------------------------------------------------------------ features
def test_num_fbank_frames_matches_kaldi_and_subsampling_formula():
    for n in [400, 401, 559, 560, 16000, 48123]:
        feats = F.compute_fbank(torch.randn(n) * 0.1)
        assert feats.shape == (F.num_fbank_frames(n), 80), n
    assert F.num_fbank_frames(399) == 0
    assert F.subsampled_length(0) == 0 and F.subsampled_length(2) == 0


def test_speed_perturb_changes_length_and_spec_aug_returns_a_masked_copy():
    wave = torch.randn(16000)
    assert abs(F.speed_perturb(wave, [1.1]).numel() - 16000 / 1.1) <= 2
    assert F.speed_perturb(wave, [1.0]).numel() == 16000 and F.speed_perturb(wave, []).numel() == 16000

    feats = torch.ones(200, 80)
    out = F.spec_aug(feats, num_t_mask=1, num_f_mask=1, max_t=20, max_f=10)
    assert feats.sum() == 200 * 80, "input must not be modified in place"
    assert out.sum() < feats.sum() and out.shape == feats.shape


# ------------------------------------------------------------------------------- dataset
def _dataset(tiny, split, train, **kw):
    tok = BPETokenizer(tiny["tokenizer"])
    return VimdDataset(tiny["root"], "manifest.tsv", split, tok, FEAT_CFG, train=train, **kw), tok


def test_dataset_item_matches_manifest(tiny_dataset):
    ds, tok = _dataset(tiny_dataset, "valid", False)
    assert len(ds) == 6
    for i in range(len(ds)):
        item = ds[i]
        assert abs(item["feats"].size(0) - ds.num_frames[i]) <= 1  # estimate used for batching
        assert item["feats"].size(1) == 80 and item["feats"].dtype == torch.float32
        assert item["tokens"].tolist() == tok.encode(item["text"])
        assert item["region_id"] == ["North", "Central", "South"].index(ds.rows[i]["region"])


def test_dropping_only_ever_applies_to_the_train_split(tiny_dataset):
    train, _ = _dataset(tiny_dataset, "train", True, max_dur=2.0)
    valid, _ = _dataset(tiny_dataset, "valid", False, max_dur=2.0)  # same limit, but valid/test are never filtered
    assert 0 < len(train) < 24 and all(d <= 2.0 for d in train.durations)
    assert len(valid) == 6
    assert any(d > 2.0 for d in valid.durations) or len(valid) == 6


def test_dataset_rejects_wrong_sample_rate_and_empty_split(tiny_dataset, tmp_path):
    import soundfile as sf

    ds, _ = _dataset(tiny_dataset, "valid", False)
    sf.write(tiny_dataset["root"] / ds.rows[0]["wav"], torch.zeros(8000).numpy(), 8000, subtype="PCM_16")
    with pytest.raises(ValueError, match="16 kHz"):
        ds[0]
    tok = BPETokenizer(tiny_dataset["tokenizer"])
    with pytest.raises(ValueError, match="no rows"):
        VimdDataset(tiny_dataset["root"], "manifest.tsv", "nonexistent", tok, FEAT_CFG, train=False)


def test_ctc_required_frames_counts_repeats():
    assert ctc_required_frames([5, 6, 7]) == 3
    assert ctc_required_frames([5, 5, 5]) == 5  # a blank is needed between equal neighbours
    assert ctc_required_frames([]) == 0


# --------------------------------------------------------------------------------- collate
def test_collate_pads_and_keeps_lengths():
    items = [
        {"utt_id": "a", "feats": torch.ones(5, 80), "tokens": torch.tensor([3, 4, 5]), "region_id": 0, "text": "x"},
        {"utt_id": "b", "feats": torch.ones(3, 80), "tokens": torch.tensor([6, 7]), "region_id": 2, "text": "y"},
    ]
    b = collate(items)
    assert b["feats"].shape == (2, 5, 80) and b["feats"][1, 3:].abs().sum() == 0
    assert b["feat_lens"].tolist() == [5, 3] and b["token_lens"].tolist() == [3, 2]
    assert b["tokens"].tolist() == [[3, 4, 5], [6, 7, 0]] and b["region_ids"].tolist() == [0, 2]
    assert b["utt_ids"] == ["a", "b"] and b["texts"] == ["x", "y"]


# ---------------------------------------------------------------------------------- groups
def test_utt_groups():
    assert utt_groups("Hôm nay trời đẹp", "hôm nay trời đẹp") == set()
    assert utt_groups("Có 25 người", "có hai mươi lăm người") == {"digit"}
    assert utt_groups("Gọi UBND nhé", "gọi ubnd nhé") == {"acronym"}
    assert utt_groups("Đăng facebook", "đăng facebook") == {"foreign"}
    assert utt_groups("Zalo 4G", "zalo bốn g") == {"digit", "foreign"}
