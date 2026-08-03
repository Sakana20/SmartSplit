from __future__ import annotations

import math
from pathlib import Path

from funasr_timeline.audio import probe_audio_duration_seconds
from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.render.base import TimelineRenderer
from funasr_timeline.render.postprocess import (
    SubtitleCue,
    SubtitlePostprocessResult,
    postprocess_subtitle_cues,
)

DEFAULT_TIMELINE_FPS = 30
DEFAULT_SUBTITLE_GAP_THRESHOLD_MS = round(20 * 1000 / DEFAULT_TIMELINE_FPS)
DEFAULT_SUBTITLE_MIN_DURATION_MS = round(6 * 1000 / DEFAULT_TIMELINE_FPS)


class SrtTimelineRenderer(TimelineRenderer):
    name = "srt"
    file_extension = ".srt"

    def __init__(
        self,
        subtitle_alignment_audio: Path | None = None,
        *,
        align_first_subtitle_to_audio_start: bool = True,
        gap_threshold_ms: int = DEFAULT_SUBTITLE_GAP_THRESHOLD_MS,
        minimum_duration_ms: int = DEFAULT_SUBTITLE_MIN_DURATION_MS,
    ) -> None:
        self.subtitle_alignment_audio = subtitle_alignment_audio
        self.align_first_subtitle_to_audio_start = align_first_subtitle_to_audio_start
        self.gap_threshold_ms = gap_threshold_ms
        self.minimum_duration_ms = minimum_duration_ms

    def render(self, items: list[SentenceTimelineItem]) -> str:
        rendered, _ = self.render_with_report(items)
        return rendered

    def render_with_report(
        self, items: list[SentenceTimelineItem]
    ) -> tuple[str, SubtitlePostprocessResult]:
        renderable_items = [
            item
            for item in items
            if item.start_ms is not None
            and item.end_ms is not None
            and item.end_ms >= item.start_ms
        ]
        if not renderable_items:
            return "", postprocess_subtitle_cues(
                [],
                gap_threshold_ms=self.gap_threshold_ms,
                minimum_duration_ms=self.minimum_duration_ms,
            )

        aligned_end_ms = (
            audio_duration_ms(self.subtitle_alignment_audio)
            if self.subtitle_alignment_audio is not None
            else None
        )
        last_item = renderable_items[-1]
        if (
            aligned_end_ms is not None
            and last_item.start_ms is not None
            and aligned_end_ms < last_item.start_ms
        ):
            raise ValueError(
                "字幕对齐音频的结束时间早于最后一条字幕的开始时间："
                f"audio_end_ms={aligned_end_ms} subtitle_start_ms={last_item.start_ms}"
            )

        cues: list[SubtitleCue] = []
        for item in renderable_items:
            assert item.start_ms is not None
            assert item.end_ms is not None
            start_ms = (
                0
                if item is renderable_items[0] and self.align_first_subtitle_to_audio_start
                else item.start_ms
            )
            end_ms = (
                aligned_end_ms if item is last_item and aligned_end_ms is not None else item.end_ms
            )

            cues.append(
                SubtitleCue(
                    source_index=item.index,
                    text=item.text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

        result = postprocess_subtitle_cues(
            cues,
            gap_threshold_ms=self.gap_threshold_ms,
            minimum_duration_ms=self.minimum_duration_ms,
        )
        blocks: list[str] = []
        for cue_index, cue in enumerate(result.cues, start=1):
            blocks.append(
                "\n".join(
                    [
                        str(cue_index),
                        f"{format_srt_timestamp(cue.start_ms)} --> "
                        f"{format_srt_timestamp(cue.end_ms)}",
                        cue.text,
                    ]
                )
            )
        return "\n\n".join(blocks) + "\n", result


def audio_duration_ms(audio_path: Path) -> int:
    if not audio_path.exists():
        raise FileNotFoundError(f"字幕对齐音频不存在：{audio_path}")
    try:
        duration_seconds = probe_audio_duration_seconds(audio_path)
    except RuntimeError as error:
        raise ValueError(f"无法读取字幕对齐音频时长：{audio_path}") from error
    frame_index = math.ceil(duration_seconds * DEFAULT_TIMELINE_FPS - 1e-9)
    return round(frame_index * 1000 / DEFAULT_TIMELINE_FPS)


def format_srt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError(f"SRT 时间戳不能为负数：{milliseconds}")

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
