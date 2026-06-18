from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from funasr_timeline.normalization import NormalizedText, normalize_text

NO_SPLIT_START = "[[NO_SPLIT]]"
NO_SPLIT_END = "[[/NO_SPLIT]]"


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


@dataclass(frozen=True, slots=True)
class _TextBlock:
    text: str
    start: int
    end: int
    paragraph_index: int
    protected: bool


class SentenceSegmenter(Protocol):
    name: str

    def segment(self, text: str) -> SegmentationResult:
        """Split manuscript text into sentence-like segments."""


class RegexSentenceSegmenter(SentenceSegmenter):
    name = "regex"

    _sentence_pattern = re.compile(r".+?(?:[。！？!?；;]|$)", re.DOTALL)
    _punctuation_boundaries = "。！？!?；;"

    def segment(self, text: str) -> SegmentationResult:
        prepared_text, blocks = _split_text_blocks(text)
        segments: list[SentenceSegment] = []

        for block in blocks:
            if block.protected:
                _append_protected_segment(segments, block)
                continue

            for match in self._sentence_pattern.finditer(block.text):
                sentence_text = match.group(0).strip()
                if not sentence_text:
                    continue
                raw_text = match.group(0)
                char_start = block.start + match.start() + _leading_whitespace_len(raw_text)
                char_end = block.start + match.end() - _trailing_whitespace_len(raw_text)
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

        return SegmentationResult(text=prepared_text, segments=segments)


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
        prepared_text, blocks = _split_text_blocks(text)
        segments: list[SentenceSegment] = []

        for block in blocks:
            if block.protected:
                _append_protected_segment(segments, block)
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
                            char_start=char_start + _leading_whitespace_len(raw_chunk),
                            char_end=char_end - _trailing_whitespace_len(raw_chunk),
                            boundary=boundary,
                        )
                    )

        return SegmentationResult(text=prepared_text, segments=segments)


SEGMENTER_FACTORIES: dict[str, Callable[[], SentenceSegmenter]] = {
    RegexSentenceSegmenter.name: RegexSentenceSegmenter,
    JiebaSubtitleSegmenter.name: JiebaSubtitleSegmenter,
}


def available_segmenters() -> tuple[str, ...]:
    return tuple(sorted(SEGMENTER_FACTORIES))


def create_segmenter(name: str) -> SentenceSegmenter:
    try:
        return SEGMENTER_FACTORIES[name]()
    except KeyError as error:
        available = "、".join(available_segmenters())
        raise ValueError(f"未知分句实现：{name}。可选值：{available}") from error


def segment_manuscript_text(text: str, segmenter: SentenceSegmenter) -> SegmentationResult:
    return segmenter.segment(text)


def export_editable_segments(segments: list[SentenceSegment]) -> str:
    lines: list[str] = []
    previous_paragraph: int | None = None
    for segment in segments:
        if previous_paragraph is not None and segment.paragraph_index != previous_paragraph:
            lines.append("")
        lines.append(segment.text)
        previous_paragraph = segment.paragraph_index
    return "\n".join(lines) + ("\n" if lines else "")


def load_editable_segments(path: Path) -> SegmentationResult:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    segments: list[SentenceSegment] = []
    text_parts: list[str] = []
    offset = 0
    paragraph_index = 0
    has_segment_in_current_paragraph = False

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            if has_segment_in_current_paragraph:
                paragraph_index += 1
                has_segment_in_current_paragraph = False
            continue

        if text_parts and not has_segment_in_current_paragraph:
            text_parts.append("\n")
            offset += 1

        char_start = offset
        text_parts.append(line)
        offset += len(line)
        segments.append(
            SentenceSegment(
                index=len(segments),
                text=line,
                paragraph_index=paragraph_index,
                char_start=char_start,
                char_end=offset,
                boundary="editable",
            )
        )
        has_segment_in_current_paragraph = True

    return SegmentationResult(text="".join(text_parts), segments=segments)


def attach_normalized_ranges(
    segments: list[SentenceSegment], normalized: NormalizedText
) -> list[SentenceSegment]:
    indexed_segments: list[SentenceSegment] = []
    for segment in segments:
        normalized_indexes: list[int] = []
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


def _append_protected_segment(segments: list[SentenceSegment], block: _TextBlock) -> None:
    sentence_text = block.text.strip()
    if not sentence_text:
        return
    segments.append(
        SentenceSegment(
            index=len(segments),
            text=sentence_text,
            paragraph_index=block.paragraph_index,
            char_start=block.start + _leading_whitespace_len(block.text),
            char_end=block.end - _trailing_whitespace_len(block.text),
            boundary="protected",
        )
    )


def _split_text_blocks(text: str) -> tuple[str, list[_TextBlock]]:
    prepared_parts: list[str] = []
    raw_position = 0
    clean_position = 0
    paragraph_index = -1
    blocks: list[_TextBlock] = []

    while raw_position < len(text):
        start_marker = text.find(NO_SPLIT_START, raw_position)
        if start_marker == -1:
            clean_position, paragraph_index = _append_unprotected_blocks(
                text[raw_position:],
                prepared_parts,
                blocks,
                clean_position,
                paragraph_index,
            )
            break

        clean_position, paragraph_index = _append_unprotected_blocks(
            text[raw_position:start_marker],
            prepared_parts,
            blocks,
            clean_position,
            paragraph_index,
        )
        protected_start = start_marker + len(NO_SPLIT_START)
        end_marker = text.find(NO_SPLIT_END, protected_start)
        if end_marker == -1:
            raise ValueError(f"缺少不分句结束标记：{NO_SPLIT_END}")

        protected_text = text[protected_start:end_marker]
        if protected_text:
            if paragraph_index < 0:
                paragraph_index = 0
            start = clean_position
            prepared_parts.append(protected_text)
            clean_position += len(protected_text)
            blocks.append(
                _TextBlock(
                    text=protected_text,
                    start=start,
                    end=clean_position,
                    paragraph_index=paragraph_index,
                    protected=True,
                )
            )
        raw_position = end_marker + len(NO_SPLIT_END)

    return "".join(prepared_parts), blocks


def _append_unprotected_blocks(
    text: str,
    prepared_parts: list[str],
    blocks: list[_TextBlock],
    clean_position: int,
    paragraph_index: int,
) -> tuple[int, int]:
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content) :]
        line_start = clean_position
        continues_existing_paragraph = bool(prepared_parts) and not _last_part_ends_with_newline(
            prepared_parts
        )
        prepared_parts.append(content)
        clean_position += len(content)

        if content.strip():
            if not continues_existing_paragraph:
                paragraph_index += 1
            blocks.append(
                _TextBlock(
                    text=content,
                    start=line_start,
                    end=clean_position,
                    paragraph_index=paragraph_index,
                    protected=False,
                )
            )

        if newline:
            prepared_parts.append(newline)
            clean_position += len(newline)

    return clean_position, paragraph_index


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
        stripped_start = clause_start + _leading_whitespace_len(clause_text)
        stripped_end = clause_end - _trailing_whitespace_len(clause_text)
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


def _leading_whitespace_len(text: str) -> int:
    return len(text) - len(text.lstrip())


def _trailing_whitespace_len(text: str) -> int:
    return len(text) - len(text.rstrip())


def _last_part_ends_with_newline(parts: list[str]) -> bool:
    for part in reversed(parts):
        if part:
            return part.endswith(("\n", "\r"))
    return False
