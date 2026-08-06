from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from funasr_timeline.audio import MediaStreamTiming, probe_subtitle_alignment_timing
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
            empty_result = postprocess_subtitle_cues(
                [],
                gap_threshold_ms=self.gap_threshold_ms,
                minimum_duration_ms=self.minimum_duration_ms,
            )
            return "", replace(
                empty_result,
                end_alignment={
                    "enabled": self.subtitle_alignment_audio is not None,
                    "applied": False,
                    "media_path": (
                        str(self.subtitle_alignment_audio)
                        if self.subtitle_alignment_audio is not None
                        else None
                    ),
                    "reason": "no_renderable_cues",
                },
            )

        alignment_timing = (
            _probe_alignment_media(self.subtitle_alignment_audio)
            if self.subtitle_alignment_audio is not None
            else None
        )
        aligned_end_ms = alignment_timing.end_milliseconds if alignment_timing is not None else None
        last_item = renderable_items[-1]
        if (
            aligned_end_ms is not None
            and last_item.start_ms is not None
            and aligned_end_ms < last_item.start_ms
        ):
            raise ValueError(
                "字幕对齐媒体的结束时间早于最后一条字幕的开始时间："
                f"media_end_ms={aligned_end_ms} subtitle_start_ms={last_item.start_ms}"
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
        end_alignment: dict[str, object] = {
            "enabled": alignment_timing is not None,
            "applied": alignment_timing is not None,
            "media_path": (
                str(self.subtitle_alignment_audio)
                if self.subtitle_alignment_audio is not None
                else None
            ),
            "source_index": last_item.index,
            "original_last_cue_end_ms": last_item.end_ms,
            "rendered_last_cue_end_ms": result.cues[-1].end_ms,
        }
        if alignment_timing is not None:
            end_alignment.update(alignment_timing.to_report())
        result = replace(result, end_alignment=end_alignment)
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
    return _probe_alignment_media(audio_path).end_milliseconds


def _probe_alignment_media(media_path: Path) -> MediaStreamTiming:
    try:
        return probe_subtitle_alignment_timing(media_path)
    except RuntimeError as error:
        raise ValueError(f"无法读取字幕对齐媒体时长：{media_path}") from error


def format_srt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError(f"SRT 时间戳不能为负数：{milliseconds}")

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
