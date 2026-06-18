from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedChar:
    normalized_index: int
    original_index: int
    original_char: str
    normalized_char: str


@dataclass(frozen=True, slots=True)
class NormalizedText:
    text: str
    chars: list[NormalizedChar]
    original_to_normalized: dict[int, list[int]]


def normalize_text(text: str) -> NormalizedText:
    chars: list[NormalizedChar] = []
    original_to_normalized: dict[int, list[int]] = {}

    for original_index, original_char in enumerate(text):
        for normalized_char in _normalize_char(original_char):
            if _should_skip(normalized_char):
                continue
            normalized_index = len(chars)
            chars.append(
                NormalizedChar(
                    normalized_index=normalized_index,
                    original_index=original_index,
                    original_char=original_char,
                    normalized_char=normalized_char,
                )
            )
            original_to_normalized.setdefault(original_index, []).append(normalized_index)

    return NormalizedText(
        text="".join(char.normalized_char for char in chars),
        chars=chars,
        original_to_normalized=original_to_normalized,
    )


def _normalize_char(char: str) -> str:
    return unicodedata.normalize("NFKC", char).lower()


def _should_skip(char: str) -> bool:
    if char.isspace():
        return True
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")
