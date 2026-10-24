import random

import numpy as np
import pytest

from teda.engine.evaluator import bootstrap_ci, error_counts, levenshtein, paired_bootstrap, rate, summarize


def naive_levenshtein(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1][-1]


def test_levenshtein_matches_textbook_dp_on_random_sequences():
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randrange(4) for _ in range(rng.randint(0, 14))]
        b = [rng.randrange(4) for _ in range(rng.randint(0, 14))]
        assert levenshtein(a, b) == naive_levenshtein(a, b), (a, b)
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert levenshtein([], [1, 2]) == 2 and levenshtein([1, 2], []) == 2 and levenshtein([], []) == 0


def test_wer_is_corpus_level_not_the_mean_of_utterance_rates():
    refs, hyps = ["a b c d", "x"], ["a b c d", "p q r"]
    c = error_counts(refs, hyps)
    assert c["err_w"].tolist() == [0, 3] and c["n_w"].tolist() == [4, 1]
    assert rate(c["err_w"], c["n_w"]) == pytest.approx(3 / 5)  # corpus level
    assert np.mean(c["err_w"] / c["n_w"]) == pytest.approx(1.5)  # what we deliberately do NOT report


def test_wer_can_exceed_one_and_cer_counts_characters():
    c = error_counts(["a"], ["p q r"])
    assert rate(c["err_w"], c["n_w"]) == 3.0
    c = error_counts(["abc"], ["abd"])
    assert c["err_c"].tolist() == [1] and c["n_c"].tolist() == [3]
    c = error_counts(["xin chào"], ["xin chao"])  # one wrong syllable, one wrong character
    assert c["err_w"].tolist() == [1] and c["err_c"].tolist() == [1]


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    n = rng.integers(5, 30, size=400)
    err = rng.binomial(n, 0.2)
    lo, hi = bootstrap_ci(err, n, n_boot=500, seed=1)
    assert lo <= rate(err, n) <= hi and hi - lo < 0.1
    assert (lo, hi) == bootstrap_ci(err, n, n_boot=500, seed=1)
    assert bootstrap_ci(np.zeros(50, dtype=int), np.full(50, 10), 100) == (0.0, 0.0)


def test_paired_bootstrap_detects_a_real_difference_and_a_null_one():
    rng = np.random.default_rng(1)
    n = np.full(600, 20)
    base = rng.binomial(n, 0.30)
    better = np.maximum(base - rng.binomial(n, 0.05), 0)  # a is better on (almost) every utterance
    res = paired_bootstrap(better, base, n, n_boot=500)
    assert res["delta"] < 0 and res["ci_hi"] < 0 and res["p_a_not_better"] < 0.01
    same = paired_bootstrap(base, base, n, n_boot=500)
    assert same["delta"] == 0 and same["ci_lo"] <= 0 <= same["ci_hi"] and same["p_a_not_better"] == 1.0


def test_summarize_breaks_down_by_region_and_group():
    recs = [
        {"utt_id": "1", "ref": "a b", "hyp": "a b", "region_id": 0, "groups": []},
        {"utt_id": "2", "ref": "c d", "hyp": "c x", "region_id": 1, "groups": ["digit"]},
        {"utt_id": "3", "ref": "e f", "hyp": "e f", "region_id": 2, "groups": ["acronym", "foreign"]},
        {"utt_id": "4", "ref": "g h", "hyp": "x y", "region_id": 2, "groups": []},
    ]
    s = summarize(recs, n_boot=50)
    assert s["n_utts"] == 4 and s["wer"] == pytest.approx(3 / 8)
    assert s["by_region"]["North"]["wer"] == 0 and s["by_region"]["Central"]["wer"] == pytest.approx(0.5)
    assert s["by_region"]["South"]["wer"] == pytest.approx(0.5) and s["by_region"]["South"]["n_utts"] == 2
    assert s["by_group"]["plain"]["n_utts"] == 2 and s["by_group"]["special"]["n_utts"] == 2
    assert s["by_group"]["digit"]["wer"] == pytest.approx(0.5) and s["by_group"]["acronym"]["wer"] == 0
    lo, hi = s["wer_ci95"]
    assert lo <= s["wer"] <= hi
    assert s["_errors"]["utt_ids"] == ["1", "2", "3", "4"]  # kept for paired comparisons between experiments
