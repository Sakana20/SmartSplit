from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from funasr_timeline.asr.base import AsrToken
from funasr_timeline.normalization import normalize_text


@dataclass(frozen=True, slots=True)
class NormalizedAsrChar:
    normalized_index: int
    token_index: int
    token_text: str
    normalized_char: str


@dataclass(frozen=True, slots=True)
class AlignmentOpcode:
    tag: str
    manuscript_start: int
    manuscript_end: int
    asr_start: int
    asr_end: int

    def to_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "manuscript_range": [self.manuscript_start, self.manuscript_end],
            "asr_range": [self.asr_start, self.asr_end],
        }


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    manuscript_text: str
    asr_text: str
    asr_chars: list[NormalizedAsrChar]
    manuscript_to_token: dict[int, int]
    opcodes: list[AlignmentOpcode]
    global_match_score: float
    unmatched_manuscript_indexes: list[int]
    unmapped_asr_indexes: list[int]

    def opcodes_as_dicts(self) -> list[dict[str, object]]:
        return [opcode.to_dict() for opcode in self.opcodes]


def normalize_asr_tokens(tokens: list[AsrToken]) -> tuple[str, list[NormalizedAsrChar]]:
    chars: list[NormalizedAsrChar] = []
    for token in tokens:
        normalized = normalize_text(token.text)
        for normalized_char in normalized.text:
            chars.append(
                NormalizedAsrChar(
                    normalized_index=len(chars),
                    token_index=token.index,
                    token_text=token.text,
                    normalized_char=normalized_char,
                )
            )
    return "".join(char.normalized_char for char in chars), chars


def align_texts(manuscript_text: str, tokens: list[AsrToken]) -> AlignmentResult:
    asr_text, asr_chars = normalize_asr_tokens(tokens)
    matcher = SequenceMatcher(a=manuscript_text, b=asr_text, autojunk=False)

    manuscript_to_token: dict[int, int] = {}
    matched_manuscript: set[int] = set()
    matched_asr: set[int] = set()
    opcodes: list[AlignmentOpcode] = []

    for tag, manuscript_start, manuscript_end, asr_start, asr_end in matcher.get_opcodes():
        opcodes.append(
            AlignmentOpcode(
                tag=tag,
                manuscript_start=manuscript_start,
                manuscript_end=manuscript_end,
                asr_start=asr_start,
                asr_end=asr_end,
            )
        )
        if tag != "equal":
            continue
        for manuscript_index, asr_index in zip(
            range(manuscript_start, manuscript_end), range(asr_start, asr_end), strict=True
        ):
            manuscript_to_token[manuscript_index] = asr_chars[asr_index].token_index
            matched_manuscript.add(manuscript_index)
            matched_asr.add(asr_index)

    unmatched_manuscript = [
        index for index in range(len(manuscript_text)) if index not in matched_manuscript
    ]
    unmapped_asr = [index for index in range(len(asr_text)) if index not in matched_asr]
    global_score = len(matched_manuscript) / len(manuscript_text) if manuscript_text else 1.0

    return AlignmentResult(
        manuscript_text=manuscript_text,
        asr_text=asr_text,
        asr_chars=asr_chars,
        manuscript_to_token=manuscript_to_token,
        opcodes=opcodes,
        global_match_score=global_score,
        unmatched_manuscript_indexes=unmatched_manuscript,
        unmapped_asr_indexes=unmapped_asr,
    )


def asr_chars_as_dicts(chars: list[NormalizedAsrChar]) -> list[dict[str, Any]]:
    return [
        {
            "normalized_index": char.normalized_index,
            "token_index": char.token_index,
            "token_text": char.token_text,
            "normalized_char": char.normalized_char,
        }
        for char in chars
    ]
