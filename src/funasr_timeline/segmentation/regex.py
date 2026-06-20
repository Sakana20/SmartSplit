from __future__ import annotations

import re

from loguru import logger

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegment, SentenceSegmenter
from funasr_timeline.segmentation.protection import (
    append_protected_segment,
    leading_whitespace_len,
    split_text_blocks,
    trailing_whitespace_len,
)


class RegexSentenceSegmenter(SentenceSegmenter):
    name = "regex"

    _sentence_pattern = re.compile(r".+?(?:[。！？!?；;]|$)", re.DOTALL)
    _punctuation_boundaries = "。！？!?；;"

    def segment(self, text: str) -> SegmentationResult:
        prepared_text, blocks = split_text_blocks(text)
        segments: list[SentenceSegment] = []
        logger.debug("regex 分句开始：blocks={} chars={}", len(blocks), len(prepared_text))

        for block in blocks:
            if block.protected:
                append_protected_segment(segments, block)
                continue

            for match in self._sentence_pattern.finditer(block.text):
                sentence_text = match.group(0).strip()
                if not sentence_text:
                    continue
                raw_text = match.group(0)
                char_start = block.start + match.start() + leading_whitespace_len(raw_text)
                char_end = block.start + match.end() - trailing_whitespace_len(raw_text)
                boundary = (
                    "punctuation"
                    if sentence_text[-1:] in self._punctuation_boundaries
                    else "paragraph"
                )
                segments.append(
                    SentenceSegment(
                        index=len(segments),
                        text=sentence_text,
                        paragraph_index=block.paragraph_index,
                        char_start=char_start,
                        char_end=char_end,
                        boundary=boundary,
                    )
                )

        logger.debug("regex 分句完成：segments={}", len(segments))
        return SegmentationResult(text=prepared_text, segments=segments)
