from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SentenceSegment:
    index: int
    text: str
    paragraph_index: int
    char_start: int
    char_end: int
    boundary: str
    normalized_text: str = ""
    normalized_start: int | None = None
    normalized_end: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "paragraph_index": self.paragraph_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "boundary": self.boundary,
            "normalized_text": self.normalized_text,
            "normalized_start": self.normalized_start,
            "normalized_end": self.normalized_end,
        }


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    text: str
    segments: list[SentenceSegment]


class SentenceSegmenter(Protocol):
    name: str

    def segment(self, text: str) -> SegmentationResult:
        """Split manuscript text into sentence-like segments."""
