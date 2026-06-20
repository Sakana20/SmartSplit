from __future__ import annotations

from dataclasses import replace

from funasr_timeline.normalization import NormalizedText
from funasr_timeline.segmentation.base import SentenceSegment


def attach_normalized_ranges(
    segments: list[SentenceSegment], normalized: NormalizedText
) -> list[SentenceSegment]:
    indexed_segments: list[SentenceSegment] = []
    for segment in segments:
        normalized_indexes: list[int] = []
        # 通过 original_to_normalized 建立句子原文范围到归一化文本范围的可复核映射。
        for original_index in range(segment.char_start, segment.char_end):
            normalized_indexes.extend(normalized.original_to_normalized.get(original_index, []))

        if normalized_indexes:
            normalized_start = min(normalized_indexes)
            normalized_end = max(normalized_indexes) + 1
            normalized_text = normalized.text[normalized_start:normalized_end]
        else:
            normalized_start = None
            normalized_end = None
            normalized_text = ""

        indexed_segments.append(
            replace(
                segment,
                normalized_text=normalized_text,
                normalized_start=normalized_start,
                normalized_end=normalized_end,
            )
        )
    return indexed_segments
