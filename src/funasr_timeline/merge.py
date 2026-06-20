from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from funasr_timeline.asr.base import AsrToken
from funasr_timeline.segmentation.base import SentenceSegment
from funasr_timeline.sentence_matching import SentenceMatchResult


@dataclass(frozen=True, slots=True)
class SentenceTimelineItem:
    index: int
    text: str
    paragraph_index: int
    start_ms: int | None
    end_ms: int | None
    duration_ms: int | None
    raw_start_ms: int | None
    raw_end_ms: int | None
    time_adjusted: bool
    match_score: float
    status: str
    matched_token_indexes: list[int]
    matched_asr_text: str
    normalized_text: str
    manuscript_char_range: tuple[int, int]
    normalized_char_range: tuple[int | None, int | None]
    asr_token_range: tuple[int | None, int | None]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "paragraph_index": self.paragraph_index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "raw_start_ms": self.raw_start_ms,
            "raw_end_ms": self.raw_end_ms,
            "time_adjusted": self.time_adjusted,
            "match_score": self.match_score,
            "status": self.status,
            "matched_token_indexes": self.matched_token_indexes,
            "matched_asr_text": self.matched_asr_text,
            "normalized_text": self.normalized_text,
            "manuscript_char_range": list(self.manuscript_char_range),
            "normalized_char_range": list(self.normalized_char_range),
            "asr_token_range": list(self.asr_token_range),
            "diagnostics": self.diagnostics,
        }


def merge_sentence_timelines(
    segments: list[SentenceSegment],
    tokens: list[AsrToken],
    matches: list[SentenceMatchResult],
) -> list[SentenceTimelineItem]:
    logger.debug(
        "合并句子时间轴开始：segments={} tokens={} matches={}",
        len(segments),
        len(tokens),
        len(matches),
    )
    token_by_index = {token.index: token for token in tokens}
    match_by_sentence_index = {match.sentence_index: match for match in matches}
    items: list[SentenceTimelineItem] = []
    previous_end_ms: int | None = None

    for segment in segments:
        match = match_by_sentence_index.get(segment.index)
        if match is None:
            items.append(_empty_item(segment, "no_match"))
            continue

        matched_token_indexes = match.matched_token_indexes
        if matched_token_indexes:
            first_token = token_by_index[matched_token_indexes[0]]
            last_token = token_by_index[matched_token_indexes[-1]]
            raw_start_ms: int | None = first_token.start_ms
            raw_end_ms: int | None = last_token.end_ms
            asr_token_range: tuple[int | None, int | None] = (
                matched_token_indexes[0],
                matched_token_indexes[-1],
            )
        else:
            raw_start_ms = None
            raw_end_ms = None
            asr_token_range = (None, None)

        start_ms = raw_start_ms
        end_ms = raw_end_ms
        time_adjusted = False
        status = match.status

        if start_ms is not None and previous_end_ms is not None and start_ms < previous_end_ms:
            logger.debug(
                "修正重叠时间：sentence_index={} raw_start_ms={} previous_end_ms={}",
                segment.index,
                start_ms,
                previous_end_ms,
            )
            start_ms = previous_end_ms
            time_adjusted = True

        if start_ms is not None and end_ms is not None:
            if end_ms < start_ms:
                status = "invalid_time_range"
                duration_ms = None
            else:
                duration_ms = end_ms - start_ms
                previous_end_ms = end_ms
        else:
            start_ms = None
            end_ms = None
            duration_ms = None

        items.append(
            SentenceTimelineItem(
                index=segment.index,
                text=segment.text,
                paragraph_index=segment.paragraph_index,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                raw_start_ms=raw_start_ms,
                raw_end_ms=raw_end_ms,
                time_adjusted=time_adjusted,
                match_score=match.match_score,
                status=status,
                matched_token_indexes=matched_token_indexes,
                matched_asr_text=match.matched_asr_text,
                normalized_text=segment.normalized_text,
                manuscript_char_range=(segment.char_start, segment.char_end),
                normalized_char_range=(segment.normalized_start, segment.normalized_end),
                asr_token_range=asr_token_range,
                diagnostics={
                    **match.diagnostics,
                    "candidate_count": match.candidate_count,
                    "selected_candidate_rank": match.selected_candidate_rank,
                },
            )
        )

    logger.debug("合并句子时间轴完成：items={}", len(items))
    return items


def _empty_item(segment: SentenceSegment, status: str) -> SentenceTimelineItem:
    return SentenceTimelineItem(
        index=segment.index,
        text=segment.text,
        paragraph_index=segment.paragraph_index,
        start_ms=None,
        end_ms=None,
        duration_ms=None,
        raw_start_ms=None,
        raw_end_ms=None,
        time_adjusted=False,
        match_score=0.0,
        status=status,
        matched_token_indexes=[],
        matched_asr_text="",
        normalized_text=segment.normalized_text,
        manuscript_char_range=(segment.char_start, segment.char_end),
        normalized_char_range=(segment.normalized_start, segment.normalized_end),
        asr_token_range=(None, None),
        diagnostics={
            "matched_chars": 0,
            "total_normalized_chars": 0,
            "unmatched_manuscript_chars": [],
            "extra_asr_tokens_nearby": [],
        },
    )
