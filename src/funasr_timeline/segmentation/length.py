from __future__ import annotations

import unicodedata


def weighted_content_half_units(text: str) -> int:
    """Measure content in half-Han-character units."""
    return sum(
        2 if _is_cjk_ideograph(char) else 1
        for char in text
        if unicodedata.category(char)[0] in {"L", "N"}
    )


def weighted_content_length(text: str) -> float:
    return weighted_content_half_units(text) / 2


def _is_cjk_ideograph(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x323AF
    )
