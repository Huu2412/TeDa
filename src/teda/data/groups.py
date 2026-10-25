"""Utterance groups for the error analysis (WER reported per group).

The corpus contains digits, acronyms and loan words whose written form is not a reliable guide to
pronunciation (see scripts/scan_text.py). Their WER is reported separately from "plain" utterances
so that gains/losses there are not hidden in the overall number.
"""

from __future__ import annotations

import re
from typing import Set

_ACRONYM = re.compile(r"(?<![\w])[A-ZĐ]{2,}(?![\w])")
_FOREIGN_LETTERS = set("fjwz")


def utt_groups(text_raw: str, text_norm: str) -> Set[str]:
    """Tags for one utterance: {'digit', 'acronym', 'foreign'} (possibly empty => 'plain')."""
    tags: Set[str] = set()
    if any(c.isdigit() for c in text_raw):
        tags.add("digit")
    if _ACRONYM.search(text_raw):
        tags.add("acronym")
    if any(c in _FOREIGN_LETTERS for c in text_norm):
        tags.add("foreign")
    return tags


def is_special(tags: Set[str]) -> bool:
    return bool(tags)
