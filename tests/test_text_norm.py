import unicodedata

import pytest

from teda.data.text_norm import (
    NumberStyle,
    Normalizer,
    int_to_words,
    tone_style_counts,
    unify_tone,
)

N = Normalizer()


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "không"),
        (5, "năm"),
        (10, "mười"),
        (11, "mười một"),
        (15, "mười lăm"),
        (21, "hai mươi mốt"),
        (24, "hai mươi bốn"),
        (25, "hai mươi lăm"),
        (100, "một trăm"),
        (105, "một trăm linh năm"),
        (110, "một trăm mười"),
        (1000, "một nghìn"),
        (1005, "một nghìn không trăm linh năm"),
        (2024, "hai nghìn không trăm hai mươi bốn"),
        (12345, "mười hai nghìn ba trăm bốn mươi lăm"),
        (1_000_000, "một triệu"),
        (1_000_005, "một triệu không trăm linh năm"),
        (2_500_000_000, "hai tỷ năm trăm triệu"),
    ],
)
def test_int_to_words(n, expected):
    assert int_to_words(n) == expected


def test_number_style_knobs():
    st = NumberStyle(linh="lẻ", four_after_tens="tư", thousand="ngàn", billion="tỉ")
    assert int_to_words(105, st) == "một trăm lẻ năm"
    assert int_to_words(24, st) == "hai mươi tư"
    assert int_to_words(1000, st) == "một ngàn"
    assert int_to_words(3_000_000_000, st) == "ba tỉ"
    with pytest.raises(ValueError):
        NumberStyle.from_dict({"nope": "x"})


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Tôi có 25 quả.", "tôi có hai mươi lăm quả"),
        ("Có 12.345 người", "có mười hai nghìn ba trăm bốn mươi lăm người"),
        ("giá 3,5 triệu", "giá ba phẩy năm triệu"),
        ("tăng 0,05", "tăng không phẩy không năm"),
        ("khoảng 50% dân số", "khoảng năm mươi phần trăm dân số"),
        ("dài 5 km", "dài năm ki lô mét"),
        ("nặng 70kg", "nặng bảy mươi ki lô gam"),
        ("giá 50.000đ", "giá năm mươi nghìn đồng"),
        ("sinh ngày 2/9/1945", "sinh ngày hai tháng chín năm một nghìn chín trăm bốn mươi lăm"),
        ("ngày 15/4", "ngày mười lăm tháng tư"),
        ("vào tháng 4", "vào tháng tư"),
        ("vào tháng 12", "vào tháng mười hai"),
        ("lúc 15h30", "lúc mười lăm giờ ba mươi phút"),
        ("lúc 8h", "lúc tám giờ"),
        ("lúc 7:05", "lúc bảy giờ không năm phút"),
        ("từ 8h-10h", "từ tám giờ đến mười giờ"),
        ("thứ 4", "thứ tư"),
        ("thứ 2", "thứ hai"),
        ("năm 2020-2021", "năm hai nghìn không trăm hai mươi đến hai nghìn không trăm hai mươi mốt"),
        ("từ 5-10kg", "từ năm đến mười ki lô gam"),
        ("gọi 0912345678", "gọi không chín một hai ba bốn năm sáu bảy tám"),
        ("Hà Nội, Sài Gòn; Đà Nẵng!", "hà nội sài gòn đà nẵng"),
        ("học-tập (tốt)", "học tập tốt"),
    ],
)
def test_normalize(raw, expected):
    assert N.normalize(raw) == expected


def test_normalize_is_nfc_and_idempotent():
    decomposed = unicodedata.normalize("NFD", "Nghiên cứu học tập")
    out = N.normalize(decomposed)
    assert out == unicodedata.normalize("NFC", out) == "nghiên cứu học tập"
    assert N.normalize(out) == out


def test_clean_for_phobert_keeps_case_and_fixes_spacing():
    assert N.clean_for_phobert("Nghiên cứu học tập,các ứng dụng.") == "Nghiên cứu học tập, các ứng dụng."
    # decimals / times are not split
    assert N.clean_for_phobert("giá 3,5 lúc 7:30") == "giá 3,5 lúc 7:30"


def test_tone_unification():
    assert unify_tone("hoà bình thuý khoẻ", "old") == "hòa bình thúy khỏe"
    assert unify_tone("hòa bình thúy khỏe", "new") == "hoà bình thuý khoẻ"
    # qu+y is never touched; closed syllables are never touched
    assert unify_tone("quý toán loá", "old") == "quý toán lóa"
    assert unify_tone("quý toán lóa", "new") == "quý toán loá"
    old, new = tone_style_counts("hòa hoà hoá thúy")
    assert (old, new) == (2, 2)
    # normalize() applies the configured style
    assert Normalizer(tone_style="new").normalize("Hòa bình") == "hoà bình"


def test_trace_reports_unresolved_items():
    trace = []
    out = N.normalize("UBND đăng Facebook 4G lúc 8h @ nhà", trace)
    kinds = {k for k, _, _ in trace}
    assert {"acronym", "foreign", "alnum", "time", "drop"} <= kinds
    assert "ubnd" in out and "facebook" in out  # utterance is kept, not dropped
    assert not any(c.isdigit() for c in out)


def test_abbreviation_map():
    n = Normalizer(abbreviations={"UBND": "ủy ban nhân dân"})
    trace = []
    assert n.normalize("Họp UBND tỉnh", trace) == "họp ủy ban nhân dân tỉnh"
    assert ("abbrev", "UBND", "ủy ban nhân dân") in trace


def test_no_residual_digits_on_odd_input():
    for raw in ["1,2,3", "1.234,56", "10.000.000.000.000", "00:99", "3.14", "1,234,567", "A4"]:
        out = N.normalize(raw)
        assert not any(c.isdigit() for c in out), (raw, out)
