from __future__ import annotations

import re
from importlib import import_module
from typing import Any, cast

from loguru import logger

from funasr_timeline.normalization import normalize_text
from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegment, SentenceSegmenter
from funasr_timeline.segmentation.protection import (
    append_protected_segment,
    leading_whitespace_len,
    split_text_blocks,
    trailing_whitespace_len,
)


class JiebaSubtitleSegmenter(SentenceSegmenter):
    name = "jieba-subtitle"

    _sentence_pattern = re.compile(r".+?(?:[。！？!?；;]|$)", re.DOTALL)
    _punctuation_boundaries = "。！？!?；;"
    _phrase_punctuation = set("，,、。！？!?；;")
    _soft_split_markers = ("特别", "直接")

    def __init__(self, max_chars: int = 10) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")
        self.max_chars = max_chars

    def segment(self, text: str) -> SegmentationResult:
        prepared_text, blocks = split_text_blocks(text)
        segments: list[SentenceSegment] = []
        logger.debug(
            "jieba-subtitle 分句开始：blocks={} chars={} max_chars={}",
            len(blocks),
            len(prepared_text),
            self.max_chars,
        )

        for block in blocks:
            if block.protected:
                append_protected_segment(segments, block)
                continue

            for sentence_match in self._sentence_pattern.finditer(block.text):
                sentence_text = sentence_match.group(0)
                if not sentence_text.strip():
                    continue
                sentence_start = block.start + sentence_match.start()
                boundary = (
                    "punctuation"
                    if sentence_text.strip()[-1:] in self._punctuation_boundaries
                    else "paragraph"
                )

                for char_start, char_end in _split_span_for_subtitle(
                    sentence_text,
                    sentence_start,
                    self.max_chars,
                    self._phrase_punctuation,
                    self._soft_split_markers,
                ):
                    raw_chunk = prepared_text[char_start:char_end]
                    chunk_text = raw_chunk.strip()
                    if not chunk_text:
                        continue
                    segments.append(
                        SentenceSegment(
                            index=len(segments),
                            text=chunk_text,
                            paragraph_index=block.paragraph_index,
                            char_start=char_start + leading_whitespace_len(raw_chunk),
                            char_end=char_end - trailing_whitespace_len(raw_chunk),
                            boundary=boundary,
                        )
                    )

        logger.debug("jieba-subtitle 分句完成：segments={}", len(segments))
        return SegmentationResult(text=prepared_text, segments=segments)


def _split_span_by_jieba(text: str, absolute_start: int, max_chars: int) -> list[tuple[int, int]]:
    jieba = cast(Any, import_module("jieba"))

    tokens = [(word, start, end) for word, start, end in jieba.tokenize(text) if word.strip()]
    chunks: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    chunk_len = 0

    for word, start, end in tokens:
        word_len = len(normalize_text(word).text)
        is_zero_width = word_len == 0

        if (
            chunk_start is not None
            and not is_zero_width
            and chunk_len > 0
            and chunk_len + word_len > max_chars
        ):
            chunks.append((absolute_start + chunk_start, absolute_start + (chunk_end or end)))
            chunk_start = None
            chunk_end = None
            chunk_len = 0

        if chunk_start is None:
            chunk_start = start
        chunk_end = end
        chunk_len += word_len

    if chunk_start is not None and chunk_end is not None:
        chunks.append((absolute_start + chunk_start, absolute_start + chunk_end))

    return chunks


def _split_span_for_subtitle(
    text: str,
    absolute_start: int,
    max_chars: int,
    phrase_punctuation: set[str],
    soft_split_markers: tuple[str, ...],
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []

    for clause_start, clause_end in _split_by_phrase_punctuation(text, phrase_punctuation):
        clause_text = text[clause_start:clause_end]
        stripped_start = clause_start + leading_whitespace_len(clause_text)
        stripped_end = clause_end - trailing_whitespace_len(clause_text)
        if stripped_start >= stripped_end:
            continue

        for phrase_start, phrase_end in _split_by_soft_markers(
            text,
            stripped_start,
            stripped_end,
            soft_split_markers,
        ):
            phrase = text[phrase_start:phrase_end]
            if len(normalize_text(phrase).text) <= max_chars:
                chunks.append((absolute_start + phrase_start, absolute_start + phrase_end))
                continue
            chunks.extend(_split_span_by_jieba(phrase, absolute_start + phrase_start, max_chars))

    return chunks


def _split_by_phrase_punctuation(text: str, punctuation: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0

    for index, char in enumerate(text):
        if char not in punctuation:
            continue
        if start < index:
            spans.append((start, index))
        start = index + 1

    if start < len(text):
        spans.append((start, len(text)))

    return spans


def _split_by_soft_markers(
    text: str,
    start: int,
    end: int,
    markers: tuple[str, ...],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = [(start, end)]

    for marker in markers:
        next_spans: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            marker_index = text.find(marker, span_start + 1, span_end)
            if marker_index == -1:
                next_spans.append((span_start, span_end))
                continue

            prefix = text[span_start:marker_index]
            suffix = text[marker_index:span_end]
            if len(normalize_text(prefix).text) < 3 or len(normalize_text(suffix).text) < 2:
                next_spans.append((span_start, span_end))
                continue

            next_spans.append((span_start, marker_index))
            next_spans.append((marker_index, span_end))
        spans = next_spans

    return spans
