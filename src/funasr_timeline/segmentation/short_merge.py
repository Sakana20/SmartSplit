from __future__ import annotations

from dataclasses import replace

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegment
from funasr_timeline.segmentation.length import weighted_content_half_units

MIN_SEGMENT_HALF_UNITS = 4


def merge_short_segments(result: SegmentationResult) -> SegmentationResult:
    """Merge isolated one-Han-character non-protected segments with neighbors."""
    segments: list[SentenceSegment] = []
    pending_short: SentenceSegment | None = None

    for segment in result.segments:
        if pending_short is not None:
            if _can_merge(pending_short, segment):
                segments.append(
                    _merge_segments(result.text, pending_short, segment, metadata=segment)
                )
                pending_short = None
                continue

            segments.append(pending_short)
            pending_short = None

        if _is_short_plain_segment(segment):
            if segments and _can_merge(segments[-1], segment):
                segments[-1] = _merge_segments(
                    result.text, segments[-1], segment, metadata=segments[-1]
                )
                continue

            pending_short = segment
            continue

        segments.append(segment)

    if pending_short is not None:
        segments.append(pending_short)

    return SegmentationResult(
        text=result.text,
        segments=[replace(segment, index=index) for index, segment in enumerate(segments)],
    )


def _is_short_plain_segment(segment: SentenceSegment) -> bool:
    if segment.boundary == "protected" or segment.segmenter == "protected":
        return False
    return 0 < weighted_content_half_units(segment.text) < MIN_SEGMENT_HALF_UNITS


def _can_merge(left: SentenceSegment, right: SentenceSegment) -> bool:
    if left.paragraph_index != right.paragraph_index:
        return False
    if left.boundary == "protected" or left.segmenter == "protected":
        return False
    if right.boundary == "protected" or right.segmenter == "protected":
        return False
    return left.char_start <= left.char_end <= right.char_start <= right.char_end


def _merge_segments(
    full_text: str,
    left: SentenceSegment,
    right: SentenceSegment,
    *,
    metadata: SentenceSegment,
) -> SentenceSegment:
    char_start = left.char_start
    char_end = right.char_end
    return replace(
        metadata,
        text=full_text[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
        normalized_text="",
        normalized_start=None,
        normalized_end=None,
    )
