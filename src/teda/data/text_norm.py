"""Vietnamese text normalisation for TeDA.

Two views of every transcript are produced:

* ``clean_for_phobert(text)``  - text for the (cached) PhoBERT teacher. Keeps case and
  punctuation (PhoBERT was trained on such text); only fixes Unicode form, tone-mark
  position and missing spaces after punctuation.
* ``normalize(text)``          - text for the CTC target. NFC, numbers/dates/times/units
  spelled out in Vietnamese, lower-case, punctuation removed, one tone-mark convention.

No utterance is ever dropped. Everything that cannot be handled deterministically
(acronyms, foreign words, letter+digit tokens, unexpected symbols) is recorded in an
optional ``trace`` so scripts/scan_text.py can report it for manual review.

Number reading in Vietnamese is not unique (linh/lẻ, bốn/tư, ...). All variants are
controlled by :class:`NumberStyle` and configs/text_norm.yaml.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Event = Tuple[str, str, str]  # (kind, source text, spoken text)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


DIGIT_WORDS = tuple(_nfc(w) for w in ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"))

VN_LETTERS = _nfc(
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmnoòóỏõọôồốổỗộơờớởỡợ"
    "pqrstuùúủũụưừứửữựvxyỳýỷỹỵ"
)
FOREIGN_LETTERS = "fjwz"  # not Vietnamese letters; kept (utterance is retained) but reported
ALLOWED = set(VN_LETTERS) | set(FOREIGN_LETTERS) | {" "}
BENIGN_PUNCT = set(".,;:!?\"'()[]{}-/\\")  # silently turned into spaces

# ----------------------------------------------------------------------------- units
UNIT_WORDS: Dict[str, str] = {
    "%": "phần trăm",
    "km/h": "ki lô mét trên giờ",
    "km2": "ki lô mét vuông", "km²": "ki lô mét vuông",
    "m2": "mét vuông", "m²": "mét vuông",
    "m3": "mét khối", "m³": "mét khối",
    "km": "ki lô mét", "cm": "xen ti mét", "mm": "mi li mét",
    "kg": "ki lô gam", "mg": "mi li gam", "ml": "mi li lít", "ha": "héc ta",
    "°C": "độ xê", "ºC": "độ xê",
    "USD": "đô la", "VND": "đồng", "VNĐ": "đồng",
    "₫": "đồng", "đ": "đồng", "g": "gam", "m": "mét", "l": "lít",
}
_SPACED_UNITS = ["km/h", "km2", "km²", "m2", "m²", "m3", "m³", "km", "cm", "mm", "kg", "mg",
                 "ml", "ha", "°C", "ºC", "USD", "VND", "VNĐ", "%"]
_TIGHT_UNITS = ["₫", "đ", "g", "m", "l"]  # single letters: only directly after the number

WEEKDAY_WORDS = {"1": "nhất", "2": "hai", "3": "ba", "4": "tư", "5": "năm", "6": "sáu", "7": "bảy"}


@dataclass(frozen=True)
class NumberStyle:
    linh: str = "linh"
    thousand: str = "nghìn"
    billion: str = "tỷ"
    four_after_tens: str = "bốn"
    one_after_tens: str = "mốt"
    five_after_tens: str = "lăm"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "NumberStyle":
        d = d or {}
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"Unknown number_style keys: {sorted(bad)}")
        return cls(**{k: _nfc(str(v)) for k, v in d.items()})


# ------------------------------------------------------------------- number reading
def _read_triple(n: int, lead: bool, st: NumberStyle) -> str:
    h, t, u = n // 100, (n // 10) % 10, n % 10
    out: List[str] = []
    if h > 0 or not lead:
        out += [DIGIT_WORDS[h], "trăm"]
    if t == 0:
        if u > 0:
            if h > 0 or not lead:
                out += [st.linh, DIGIT_WORDS[u]]
            else:
                out.append(DIGIT_WORDS[u])
    elif t == 1:
        out.append("mười")
        if u == 5:
            out.append(st.five_after_tens)
        elif u > 0:
            out.append(DIGIT_WORDS[u])
    else:
        out += [DIGIT_WORDS[t], "mươi"]
        if u == 1:
            out.append(st.one_after_tens)
        elif u == 4:
            out.append(st.four_after_tens)
        elif u == 5:
            out.append(st.five_after_tens)
        elif u > 0:
            out.append(DIGIT_WORDS[u])
    return " ".join(out)


def int_to_words(n: int, st: NumberStyle = NumberStyle()) -> str:
    """Cardinal reading for 0 <= n < 10**12."""
    if n < 0 or n >= 10**12:
        raise ValueError(f"int_to_words supports 0 <= n < 10**12, got {n}")
    if n == 0:
        return DIGIT_WORDS[0]
    scales = ["", st.thousand, "triệu", st.billion]
    groups: List[int] = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    parts: List[str] = []
    started = False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        w = _read_triple(g, not started, st)
        started = True
        parts.append(w + (" " + scales[i] if scales[i] else ""))
    return " ".join(parts)


def digits_to_words(s: str) -> str:
    return " ".join(DIGIT_WORDS[int(c)] for c in s if c.isdigit())


# ---------------------------------------------------------------------- tone marks
_MARKS = "̣̀́̉̃"
_END = r"(?![^\W_]|[̀-ͯ])"  # next char is not a letter/digit/combining mark
# NFD patterns. "mark on second vowel" = new style (hoà); "mark on first vowel" = old style (hòa).
_MARK_ON_SECOND = re.compile(r"(?<![qQ])(?P<v1>[oOuU])(?P<v2>[aAeEyY])(?P<t>[" + _MARKS + r"])" + _END)
_MARK_ON_FIRST = re.compile(r"(?<![qQ])(?P<v1>[oOuU])(?P<t>[" + _MARKS + r"])(?P<v2>[aAeEyY])" + _END)
_VALID_PAIRS = {"oa", "oe", "uy"}


def _tone_sub(pattern: re.Pattern, text_nfd: str, move_to_first: bool) -> str:
    def repl(m: re.Match) -> str:
        pair = (m.group("v1") + m.group("v2")).lower()
        if pair not in _VALID_PAIRS:
            return m.group(0)
        if move_to_first:
            return m.group("v1") + m.group("t") + m.group("v2")
        return m.group("v1") + m.group("v2") + m.group("t")

    return pattern.sub(repl, text_nfd)


def unify_tone(text: str, style: str = "old") -> str:
    """Put the tone mark of open syllables oa/oe/uy in one convention.

    old: hòa, khỏe, thúy      new: hoà, khoẻ, thuý      ("qu"+y is never touched: quý)
    """
    if style not in ("old", "new"):
        raise ValueError("tone_style must be 'old' or 'new'")
    nfd = unicodedata.normalize("NFD", text)
    if style == "old":
        nfd = _tone_sub(_MARK_ON_SECOND, nfd, move_to_first=True)   # hoà -> hòa
    else:
        nfd = _tone_sub(_MARK_ON_FIRST, nfd, move_to_first=False)   # hòa -> hoà
    return _nfc(nfd)


def tone_style_counts(text: str) -> Tuple[int, int]:
    """(# old-style, # new-style) open-syllable tone placements found in `text`."""
    nfd = unicodedata.normalize("NFD", text)
    old = sum(1 for m in _MARK_ON_FIRST.finditer(nfd) if (m.group("v1") + m.group("v2")).lower() in _VALID_PAIRS)
    new = sum(1 for m in _MARK_ON_SECOND.finditer(nfd) if (m.group("v1") + m.group("v2")).lower() in _VALID_PAIRS)
    return old, new


def load_abbreviations(path: Optional[str | Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path or not Path(path).exists():
        return mapping
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"{path}:{ln}: expected 'RAW<TAB>SPOKEN', got {line!r}")
            mapping[_nfc(parts[0].strip())] = _nfc(parts[1].strip())
    return mapping


# --------------------------------------------------------------------- the normaliser
_NUM = r"\d+(?:[.,]\d+)*"
_TIME_CORE = r"\d{1,2}(?:h|H|:)(?:\d{2})?"


def _alt(units: List[str]) -> str:
    return "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))


class Normalizer:
    def __init__(
        self,
        style: NumberStyle = NumberStyle(),
        tone_style: str = "old",
        abbreviations: Optional[Dict[str, str]] = None,
    ):
        if tone_style not in ("old", "new"):
            raise ValueError("tone_style must be 'old' or 'new'")
        self.style = style
        self.tone_style = tone_style
        self.abbreviations = dict(abbreviations or {})

        all_units = _SPACED_UNITS + _TIGHT_UNITS
        self._date_full = re.compile(
            r"(?:\b[Nn]gày\s+)?(?<![\d/.\-])(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{4})(?!\d)"
        )
        self._date_dm = re.compile(r"\b[Nn]gày\s+(?P<d>\d{1,2})[/\-](?P<m>\d{1,2})(?![\d/\-])")
        self._month_y = re.compile(r"\b[Tt]háng\s+(?P<m>\d{1,2})[/\-](?P<y>\d{4})(?!\d)")
        self._time_range = re.compile(
            r"(?<![\d:.,/\-])(?P<t1>" + _TIME_CORE + r")\s?-\s?(?P<t2>" + _TIME_CORE + r")(?![\w])"
        )
        self._time = re.compile(r"(?<![\d:.,/\-])(?P<h>\d{1,2})(?:h|H|:)(?P<m>\d{2})?(?![\w])")
        self._range = re.compile(
            r"(?<![\d.,/\-])(?P<a>\d+(?:[.,]\d+)?)\s?-\s?(?P<b>\d+(?:[.,]\d+)?)"
            r"(?:\s?(?P<u>" + _alt(all_units) + r"))?(?![\w])(?!\s?-\s?\d)"
        )
        self._unit_spaced = re.compile(r"(?P<num>" + _NUM + r")\s?(?P<u>" + _alt(_SPACED_UNITS) + r")(?![\w])")
        self._unit_tight = re.compile(r"(?P<num>" + _NUM + r")(?P<u>" + _alt(_TIGHT_UNITS) + r")(?![\w])")
        self._currency = re.compile(r"\$\s?(?P<num>" + _NUM + r")")
        self._weekday = re.compile(r"\b(?P<w>[Tt]hứ)\s+(?P<n>[1-7])(?![\d/,.:\w])")
        self._month4 = re.compile(r"\b(?P<w>[Tt]háng)\s+4(?![\d/,.:\w])")
        self._generic = re.compile(_NUM)
        self._acronym = re.compile(r"(?<![\w])[A-ZĐ]{2,}(?![\w])")
        self._space_after_punct = re.compile(r"(?<=[,;:!?])(?=[^\W\d_])")

        if self.abbreviations:
            keys = sorted(self.abbreviations, key=len, reverse=True)
            self._abbrev = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![\w])")
        else:
            self._abbrev = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, path: str | Path = "configs/text_norm.yaml") -> "Normalizer":
        import yaml

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(
            style=NumberStyle.from_dict(cfg.get("number_style")),
            tone_style=cfg.get("tone_style", "old"),
            abbreviations=load_abbreviations(cfg.get("abbreviations_file")),
        )

    # ------------------------------------------------------------------------ helpers
    @staticmethod
    def _log(trace: Optional[List[Event]], kind: str, src: str, dst: str) -> None:
        if trace is not None:
            trace.append((kind, src.strip(), dst.strip()))

    def _int_words(self, digits: str) -> str:
        return int_to_words(int(digits), self.style)

    def _number(self, s: str, trace: Optional[List[Event]]) -> str:
        """Spoken form of a numeric token such as 25, 1.000, 3,5, 0912345678."""
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            digits = s.replace(".", "")
            return self._int_or_digits(digits, "int_grouped", s, trace)
        if re.fullmatch(r"\d{1,3}(?:,\d{3}){2,}", s):  # 1,234,567 (English grouping)
            return self._int_or_digits(s.replace(",", ""), "int_grouped_en", s, trace)
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", s):  # 1.234,56
            ip, fp = s.split(",")
            return self._decimal(ip.replace(".", ""), fp, "decimal", s, trace)
        if re.fullmatch(r"\d+,\d+", s):
            ip, fp = s.split(",")
            kind = "decimal_3digit" if len(fp) == 3 else "decimal"  # 1,234 may be English thousands
            return self._decimal(ip, fp, kind, s, trace)
        if re.fullmatch(r"\d+\.\d+", s):
            ip, fp = s.split(".")
            return self._decimal(ip, fp, "decimal_dot", s, trace)
        if s.isdigit():
            if len(s) > 1 and s[0] == "0":
                out = digits_to_words(s)
                self._log(trace, "digitwise", s, out)
                return out
            if len(s) >= 10:
                out = digits_to_words(s)
                self._log(trace, "digitwise_long", s, out)
                return out
            return self._int_or_digits(s, "int", s, trace)
        parts = [p for p in re.split(r"[.,]", s) if p]  # odd mixes: read part by part
        out = " phẩy ".join(self._number(p, None) for p in parts)
        self._log(trace, "fallback", s, out)
        return out

    def _int_or_digits(self, digits: str, kind: str, src: str, trace) -> str:
        if int(digits) >= 10**12:
            out = digits_to_words(digits)
            self._log(trace, "big_int", src, out)
            return out
        out = self._int_words(digits)
        self._log(trace, kind, src, out)
        return out

    def _decimal(self, ip: str, fp: str, kind: str, src: str, trace) -> str:
        ipw = digits_to_words(ip) if (len(ip) > 1 and ip[0] == "0") else self._int_words(ip)
        fpw = digits_to_words(fp) if (fp[0] == "0" or len(fp) > 2) else self._int_words(fp)
        out = f"{ipw} phẩy {fpw}"
        self._log(trace, kind, src, out)
        return out

    def _time_words(self, h: str, m: Optional[str]) -> Optional[str]:
        hi = int(h)
        if hi > 24:
            return None
        out = f"{self._int_words(h)} giờ"
        if m and m != "00":
            mi = int(m)
            if mi >= 60:
                return None
            out += f" {digits_to_words(m) if m[0] == '0' else self._int_words(m)} phút"
        return out

    # ------------------------------------------------------------- numeric rewriting
    def _rewrite_numbers(self, t: str, trace: Optional[List[Event]]) -> str:
        def date_full(m: re.Match) -> str:
            d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
            if not (1 <= d <= 31 and 1 <= mo <= 12 and 1000 <= y <= 2999):
                return m.group(0)
            out = f" ngày {self._int_words(m.group('d'))} tháng {self._month_words(mo)} năm {self._int_words(m.group('y'))} "
            self._log(trace, "date", m.group(0), out)
            return out

        def date_dm(m: re.Match) -> str:
            d, mo = int(m.group("d")), int(m.group("m"))
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                return m.group(0)
            out = f" ngày {self._int_words(m.group('d'))} tháng {self._month_words(mo)} "
            self._log(trace, "date", m.group(0), out)
            return out

        def month_y(m: re.Match) -> str:
            mo = int(m.group("m"))
            if not 1 <= mo <= 12:
                return m.group(0)
            out = f" tháng {self._month_words(mo)} năm {self._int_words(m.group('y'))} "
            self._log(trace, "date", m.group(0), out)
            return out

        def time_range(m: re.Match) -> str:
            outs = []
            for key in ("t1", "t2"):
                mm = self._time.fullmatch(m.group(key))
                w = self._time_words(mm.group("h"), mm.group("m")) if mm else None
                if w is None:
                    return m.group(0)
                outs.append(w)
            out = f" {outs[0]} đến {outs[1]} "
            self._log(trace, "time_range", m.group(0), out)
            return out

        def time(m: re.Match) -> str:
            w = self._time_words(m.group("h"), m.group("m"))
            if w is None:
                return m.group(0)
            self._log(trace, "time", m.group(0), w)
            return f" {w} "

        def rng(m: re.Match) -> str:
            a, b, u = m.group("a"), m.group("b"), m.group("u")
            out = f" {self._number(a, None)} đến {self._number(b, None)} "
            if u:
                out += f"{UNIT_WORDS[u]} "
            self._log(trace, "range", m.group(0), out)
            return out

        def unit(m: re.Match) -> str:
            w = self._number(m.group("num"), trace)
            u = UNIT_WORDS[m.group("u")]
            self._log(trace, "unit", m.group(0), f"{w} {u}")
            return f" {w} {u} "

        def currency(m: re.Match) -> str:
            w = self._number(m.group("num"), trace)
            self._log(trace, "unit", m.group(0), f"{w} đô la")
            return f" {w} đô la "

        def weekday(m: re.Match) -> str:
            out = f"{m.group('w')} {WEEKDAY_WORDS[m.group('n')]}"
            self._log(trace, "weekday", m.group(0), out)
            return out

        def month4(m: re.Match) -> str:
            self._log(trace, "month", m.group(0), "tháng tư")
            return f"{m.group('w')} tư"

        def generic(m: re.Match) -> str:
            s = m.string
            if m.start() > 0 and s[m.start() - 1].isalpha():
                self._log(trace, "alnum", s[max(0, m.start() - 4): m.end() + 2], "")
            elif m.end() < len(s) and s[m.end()].isalpha():
                self._log(trace, "alnum", s[max(0, m.start() - 2): m.end() + 4], "")
            return f" {self._number(m.group(0), trace)} "

        t = self._date_full.sub(date_full, t)
        t = self._date_dm.sub(date_dm, t)
        t = self._month_y.sub(month_y, t)
        t = self._time_range.sub(time_range, t)
        t = self._time.sub(time, t)
        t = self._range.sub(rng, t)
        t = self._unit_spaced.sub(unit, t)
        t = self._unit_tight.sub(unit, t)
        t = self._currency.sub(currency, t)
        t = self._weekday.sub(weekday, t)
        t = self._month4.sub(month4, t)
        t = self._generic.sub(generic, t)
        return t

    def _month_words(self, month: int) -> str:
        return "tư" if month == 4 else self._int_words(str(month))

    # ---------------------------------------------------------------------- public
    @staticmethod
    def _pre_clean(t: str) -> str:
        table = {
            "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
            "‘": "'", "’": "'", "“": '"', "”": '"',
            " ": " ", "​": "", "﻿": "", "…": " ",
        }
        return t.translate({ord(k): v for k, v in table.items()})

    def clean_for_phobert(self, text: str) -> str:
        """Text for PhoBERT: keep case/punctuation, fix Unicode, tone mark, spacing."""
        t = self._pre_clean(_nfc(text))
        t = self._space_after_punct.sub(" ", t)
        t = unify_tone(t, self.tone_style)
        return " ".join(t.split())

    def normalize(self, text: str, trace: Optional[List[Event]] = None) -> str:
        """Text for the CTC target (lower-case, spoken-form numbers, no punctuation)."""
        t = self._pre_clean(_nfc(text))
        if self._abbrev is not None:
            def abbrev(m: re.Match) -> str:
                out = self.abbreviations[m.group(0)]
                self._log(trace, "abbrev", m.group(0), out)
                return f" {out} "

            t = self._abbrev.sub(abbrev, t)
        t = self._rewrite_numbers(t, trace)
        if trace is not None:
            for m in self._acronym.finditer(t):
                trace.append(("acronym", m.group(0), ""))
        t = t.lower()

        chars: List[str] = []
        for ch in t:
            if ch in ALLOWED:
                chars.append(ch)
            elif ch.isspace() or ch in BENIGN_PUNCT:
                chars.append(" ")
            else:
                chars.append(" ")
                if trace is not None:
                    trace.append(("residual_digit" if ch.isdigit() else "drop", ch, ""))
        t = unify_tone("".join(chars), self.tone_style)
        t = " ".join(t.split())
        if trace is not None:
            for tok in t.split():
                if any(c in FOREIGN_LETTERS for c in tok):
                    trace.append(("foreign", tok, ""))
        return t
