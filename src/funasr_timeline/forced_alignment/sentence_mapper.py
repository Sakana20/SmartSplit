from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from loguru import logger

from funasr_timeline.forced_alignment.base import ForcedAlignmentResult, ForcedAlignmentUnit
from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.segmentation.base import SentenceSegment


@dataclass(frozen=True, slots=True)
class ForcedSentenceTiming:
    sentence_index: int
    start_ms: int | None
    end_ms: int | None
    raw_start_ms: int | None
    raw_end_ms: int | None
    time_adjusted: bool
    status: str
    unit_range: tuple[int | None, int | None]
    unit_indexes: list[int]
    matched_text: str
    diagnostics: dict[str, Any]

    @property
    def duration_ms(self) -> int | None:
        if self.start_ms is None or self.end_ms is None:
            return None
        if self.end_ms < self.start_ms:
            return None
        return self.end_ms - self.start_ms

    def to_telemetry(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "raw_start_ms": self.raw_start_ms,
            "raw_end_ms": self.raw_end_ms,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "unit_range": list(self.unit_range),
            "unit_indexes": self.unit_indexes,
            "matched_text": self.matched_text,
            "time_adjusted": self.time_adjusted,
            "diagnostics": self.diagnostics,
        }


def map_forced_alignment_to_sentence_items(
    segments: list[SentenceSegment],
    forced_result: ForcedAlignmentResult,
    timeline_provider: str,
) -> tuple[list[SentenceTimelineItem], list[ForcedSentenceTiming]]:
    timings = map_forced_alignment_to_sentence_timings(segments, forced_result)
    timing_by_sentence_index = {timing.sentence_index: timing for timing in timings}
    items: list[SentenceTimelineItem] = []

    for segment in segments:
        timing = timing_by_sentence_index.get(segment.index)
        if timing is None:
            timing = _empty_timing(segment, "forced_missing_unit")
        items.append(_timeline_item_from_forced(segment, timing, timeline_provider))

    return items, timings


def map_forced_alignment_to_sentence_timings(
    segments: list[SentenceSegment],
    forced_result: ForcedAlignmentResult,
) -> list[ForcedSentenceTiming]:
    logger.debug(
        "forced alignment 映射到分句开始：segments={} units={}",
        len(segments),
        len(forced_result.units),
    )
    normalized_char_to_unit = _normalized_char_to_unit_indexes(forced_result.units)
    text_mismatch_diagnostics = (
        {}
        if forced_result.normalized_text_match
        else {
            "forced_text_mismatch": True,
            "text_similarity": round(
                SequenceMatcher(
                    a=forced_result.normalized_text,
                    b=forced_result.forced_normalized_text,
                    autojunk=False,
                ).ratio(),
                6,
            ),
            "opcodes": _text_opcodes(
                forced_result.normalized_text,
                forced_result.forced_normalized_text,
            ),
        }
    )

    timings: list[ForcedSentenceTiming] = []
    previous_end_ms: int | None = None
    for segment in segments:
        timing = _map_segment(
            segment=segment,
            forced_result=forced_result,
            normalized_char_to_unit=normalized_char_to_unit,
            text_mismatch_diagnostics=text_mismatch_diagnostics,
            previous_end_ms=previous_end_ms,
        )
        timings.append(timing)
        if timing.end_ms is not None:
            previous_end_ms = timing.end_ms

    logger.debug("forced alignment 映射到分句完成：timings={}", len(timings))
    return timings


def _map_segment(
    segment: SentenceSegment,
    forced_result: ForcedAlignmentResult,
    normalized_char_to_unit: list[int],
    text_mismatch_diagnostics: dict[str, Any],
    previous_end_ms: int | None,
) -> ForcedSentenceTiming:
    if not segment.normalized_text:
        return _empty_timing(segment, "forced_empty_segment")
    if segment.normalized_start is None or segment.normalized_end is None:
        return _empty_timing(segment, "forced_missing_unit")

    char_start = segment.normalized_start
    char_end = segment.normalized_end
    if char_start < 0 or char_end > len(normalized_char_to_unit) or char_start >= char_end:
        return _empty_timing(
            segment,
            "forced_missing_unit",
            diagnostics={
                **text_mismatch_diagnostics,
                "requested_normalized_range": [char_start, char_end],
                "forced_normalized_chars": len(normalized_char_to_unit),
            },
        )

    unit_indexes = _unique_in_order(normalized_char_to_unit[char_start:char_end])
    if not unit_indexes:
        return _empty_timing(segment, "forced_missing_unit", text_mismatch_diagnostics)

    unit_by_index = {unit.index: unit for unit in forced_result.units}
    first_unit = unit_by_index[unit_indexes[0]]
    last_unit = unit_by_index[unit_indexes[-1]]
    raw_start_ms: int | None = first_unit.start_ms
    raw_end_ms: int | None = last_unit.end_ms
    start_ms = raw_start_ms
    end_ms = raw_end_ms
    status = "forced_ok"
    time_adjusted = False

    if start_ms is not None and previous_end_ms is not None and start_ms < previous_end_ms:
        start_ms = previous_end_ms
        time_adjusted = True

    if start_ms is None or end_ms is None:
        status = "forced_missing_unit"
        start_ms = None
        end_ms = None
    elif end_ms < start_ms:
        status = "forced_invalid_time_range"

    matched_units = [unit_by_index[index] for index in unit_indexes]
    return ForcedSentenceTiming(
        sentence_index=segment.index,
        start_ms=start_ms,
        end_ms=end_ms,
        raw_start_ms=raw_start_ms,
        raw_end_ms=raw_end_ms,
        time_adjusted=time_adjusted,
        status=status,
        unit_range=(unit_indexes[0], unit_indexes[-1]),
        unit_indexes=unit_indexes,
        matched_text="".join(unit.text for unit in matched_units),
        diagnostics={
            **text_mismatch_diagnostics,
            "matched_normalized_chars": char_end - char_start,
            "total_normalized_chars": len(segment.normalized_text),
            "normalized_char_range": [char_start, char_end],
        },
    )


def _timeline_item_from_forced(
    segment: SentenceSegment,
    timing: ForcedSentenceTiming,
    timeline_provider: str,
) -> SentenceTimelineItem:
    return SentenceTimelineItem(
        index=segment.index,
        text=segment.text,
        paragraph_index=segment.paragraph_index,
        start_ms=timing.start_ms,
        end_ms=timing.end_ms,
        duration_ms=timing.duration_ms,
        raw_start_ms=timing.raw_start_ms,
        raw_end_ms=timing.raw_end_ms,
        time_adjusted=timing.time_adjusted,
        match_score=1.0 if timing.status == "forced_ok" else 0.0,
        status="ok" if timing.status == "forced_ok" else timing.status,
        matched_token_indexes=[],
        matched_asr_text="",
        normalized_text=segment.normalized_text,
        manuscript_char_range=(segment.char_start, segment.char_end),
        normalized_char_range=(segment.normalized_start, segment.normalized_end),
        asr_token_range=(None, None),
        diagnostics={
            "timeline_provider": timeline_provider,
            "primary_timing_source": "qwen3-forced",
            "forced_status": timing.status,
            "forced_unit_range": list(timing.unit_range),
            "forced_unit_indexes": timing.unit_indexes,
            **timing.diagnostics,
        },
    )


def _empty_timing(
    segment: SentenceSegment,
    status: str,
    diagnostics: dict[str, Any] | None = None,
) -> ForcedSentenceTiming:
    return ForcedSentenceTiming(
        sentence_index=segment.index,
        start_ms=None,
        end_ms=None,
        raw_start_ms=None,
        raw_end_ms=None,
        time_adjusted=False,
        status=status,
        unit_range=(None, None),
        unit_indexes=[],
        matched_text="",
        diagnostics={
            "matched_normalized_chars": 0,
            "total_normalized_chars": len(segment.normalized_text),
            **(diagnostics or {}),
        },
    )


def _normalized_char_to_unit_indexes(units: list[ForcedAlignmentUnit]) -> list[int]:
    mapping: list[int] = []
    for unit in units:
        mapping.extend([unit.index] * len(unit.normalized_text))
    return mapping


def _text_opcodes(left: str, right: str) -> list[dict[str, int | str]]:
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    return [
        {
            "tag": tag,
            "manuscript_start": left_start,
            "manuscript_end": left_end,
            "forced_start": right_start,
            "forced_end": right_end,
        }
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes()
    ]


def _unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
