from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any

from loguru import logger

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegment, SentenceSegmenter
from funasr_timeline.segmentation.length import weighted_content_half_units
from funasr_timeline.segmentation.protection import (
    append_protected_segment,
    split_text_blocks,
)
from funasr_timeline.segmentation.short_merge import merge_short_segments

DEFAULT_THRESHOLD = 10
PHRASE_BOUNDARIES = frozenset("，,、。！？!?；;：:")


class HanlpSegmenter(SentenceSegmenter):
    """Split text on HanLP constituency tokens without breaking a token."""

    name = "hanlp"

    def __init__(self, threshold: int = DEFAULT_THRESHOLD) -> None:
        if threshold <= 0:
            raise ValueError("threshold 必须大于 0")
        self.threshold = threshold

    def segment(self, text: str) -> SegmentationResult:
        prepared_text, blocks = split_text_blocks(text)
        segments: list[SentenceSegment] = []
        logger.debug(
            "hanlp 分句开始：blocks={} chars={} threshold={}",
            len(blocks),
            len(prepared_text),
            self.threshold,
        )

        for block in blocks:
            if block.protected:
                append_protected_segment(segments, block)
                continue
            if not block.text.strip():
                continue

            for sentence_start, sentence_end in _phrase_spans(block.text):
                sentence_text = block.text[sentence_start:sentence_end]
                if weighted_content_half_units(sentence_text) == 0:
                    continue

                parser = _load_constituency_parser()
                document = parser(sentence_text, tasks="con")
                tokens = [str(token) for token in document["con"].leaves()]
                for chunk_start, chunk_end in _chunk_token_spans(
                    sentence_text, tokens, self.threshold
                ):
                    raw_chunk = sentence_text[chunk_start:chunk_end]
                    content_start, content_end = _trim_non_content_edges(raw_chunk)
                    if content_start >= content_end:
                        continue
                    char_start = block.start + sentence_start + chunk_start + content_start
                    char_end = block.start + sentence_start + chunk_start + content_end
                    segments.append(
                        SentenceSegment(
                            index=len(segments),
                            text=prepared_text[char_start:char_end],
                            paragraph_index=block.paragraph_index,
                            char_start=char_start,
                            char_end=char_end,
                            boundary="threshold",
                            segmenter=self.name,
                        )
                    )

        result = merge_short_segments(SegmentationResult(text=prepared_text, segments=segments))
        logger.debug("hanlp 分句完成：segments={}", len(result.segments))
        return result


@lru_cache(maxsize=1)
def _load_constituency_parser() -> Any:
    hanlp = import_module("hanlp")
    model = hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
    logger.info("加载 HanLP constituency 模型：{}", model)
    return hanlp.load(model)


def _chunk_token_spans(text: str, tokens: list[str], threshold: int) -> list[tuple[int, int]]:
    token_spans = _locate_tokens(text, tokens)
    chunks: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    chunk_length = 0

    for token, (token_start, token_end) in zip(tokens, token_spans, strict=True):
        token_length = weighted_content_half_units(token)
        if chunk_start is not None and token_length and chunk_length + token_length > threshold * 2:
            if chunk_end is not None:
                chunks.append((chunk_start, chunk_end))
            chunk_start = None
            chunk_end = None
            chunk_length = 0

        if chunk_start is None:
            chunk_start = token_start
        chunk_end = token_end
        chunk_length += token_length

    if chunk_start is not None and chunk_end is not None:
        chunks.append((chunk_start, chunk_end))
    return chunks


def _locate_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start == -1:
            raise ValueError(f"HanLP token 无法映射回原文：{token!r}，起始位置 {cursor}")
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def _trim_non_content_edges(text: str) -> tuple[int, int]:
    start = 0
    end = len(text)
    while start < end and weighted_content_half_units(text[start]) == 0:
        start += 1
    while end > start and weighted_content_half_units(text[end - 1]) == 0:
        end -= 1
    return start, end


def _phrase_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char not in PHRASE_BOUNDARIES:
            continue
        spans.append((start, index + 1))
        start = index + 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans
