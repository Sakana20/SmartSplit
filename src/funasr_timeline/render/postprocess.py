from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    source_index: int
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class SubtitlePostprocessResult:
    cues: list[SubtitleCue]
    adjustments: list[dict[str, Any]]
    unresolved_short_cues: list[dict[str, int]]
    end_alignment: dict[str, Any] = field(default_factory=dict)

    def to_report(self, *, gap_threshold_ms: int, minimum_duration_ms: int) -> dict[str, Any]:
        return {
            "config": {
                "gap_threshold_ms": gap_threshold_ms,
                "minimum_duration_ms": minimum_duration_ms,
            },
            "adjustment_count": len(self.adjustments),
            "adjustments": self.adjustments,
            "unresolved_short_cues": self.unresolved_short_cues,
            "end_alignment": self.end_alignment,
        }


def postprocess_subtitle_cues(
    cues: list[SubtitleCue],
    *,
    gap_threshold_ms: int,
    minimum_duration_ms: int,
) -> SubtitlePostprocessResult:
    """Repair short blank gaps first, then extend unusually short cues into free time."""
    if gap_threshold_ms < 0:
        raise ValueError("字幕间隙阈值不能为负数")
    if minimum_duration_ms < 0:
        raise ValueError("字幕最短持续时间不能为负数")

    processed = list(cues)
    adjustments: list[dict[str, Any]] = []

    for index in range(len(processed) - 1):
        current = processed[index]
        following = processed[index + 1]
        gap_ms = following.start_ms - current.end_ms
        if 0 < gap_ms <= gap_threshold_ms:
            processed[index] = replace(current, end_ms=following.start_ms)
            adjustments.append(
                {
                    "type": "short_gap_filled",
                    "source_index": current.source_index,
                    "next_source_index": following.source_index,
                    "gap_ms": gap_ms,
                    "original_end_ms": current.end_ms,
                    "adjusted_end_ms": following.start_ms,
                }
            )

    for index, cue in enumerate(processed):
        duration_ms = cue.end_ms - cue.start_ms
        if duration_ms >= minimum_duration_ms:
            continue

        original_start_ms = cue.start_ms
        original_end_ms = cue.end_ms
        missing_ms = minimum_duration_ms - duration_ms

        if index + 1 < len(processed):
            right_limit_ms = processed[index + 1].start_ms
            right_extension_ms = min(missing_ms, max(0, right_limit_ms - cue.end_ms))
            cue = replace(cue, end_ms=cue.end_ms + right_extension_ms)
            missing_ms -= right_extension_ms

        if missing_ms > 0:
            left_limit_ms = processed[index - 1].end_ms if index > 0 else 0
            left_extension_ms = min(missing_ms, max(0, cue.start_ms - left_limit_ms))
            cue = replace(cue, start_ms=cue.start_ms - left_extension_ms)

        processed[index] = cue
        if cue.start_ms != original_start_ms or cue.end_ms != original_end_ms:
            adjustments.append(
                {
                    "type": "short_cue_extended",
                    "source_index": cue.source_index,
                    "original_start_ms": original_start_ms,
                    "original_end_ms": original_end_ms,
                    "adjusted_start_ms": cue.start_ms,
                    "adjusted_end_ms": cue.end_ms,
                }
            )

    unresolved = [
        {
            "source_index": cue.source_index,
            "start_ms": cue.start_ms,
            "end_ms": cue.end_ms,
            "duration_ms": cue.end_ms - cue.start_ms,
        }
        for cue in processed
        if cue.end_ms - cue.start_ms < minimum_duration_ms
    ]
    return SubtitlePostprocessResult(
        cues=processed,
        adjustments=adjustments,
        unresolved_short_cues=unresolved,
    )
