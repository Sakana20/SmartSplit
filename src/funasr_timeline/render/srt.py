from __future__ import annotations

import math
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.render.base import TimelineRenderer

DEFAULT_TIMELINE_FPS = 30


class SrtTimelineRenderer(TimelineRenderer):
    name = "srt"
    file_extension = ".srt"

    def __init__(self, subtitle_alignment_audio: Path | None = None) -> None:
        self.subtitle_alignment_audio = subtitle_alignment_audio

    def render(self, items: list[SentenceTimelineItem]) -> str:
        renderable_items = [
            item
            for item in items
            if item.start_ms is not None
            and item.end_ms is not None
            and item.end_ms >= item.start_ms
        ]
        if not renderable_items:
            return ""

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

        blocks: list[str] = []
        for cue_index, item in enumerate(renderable_items, start=1):
            assert item.start_ms is not None
            assert item.end_ms is not None
            end_ms = (
                aligned_end_ms if item is last_item and aligned_end_ms is not None else item.end_ms
            )

            blocks.append(
                "\n".join(
                    [
                        str(cue_index),
                        f"{format_srt_timestamp(item.start_ms)} --> {format_srt_timestamp(end_ms)}",
                        item.text,
                    ]
                )
            )
        return "\n\n".join(blocks) + "\n"


def audio_duration_ms(audio_path: Path) -> int:
    if not audio_path.exists():
        raise FileNotFoundError(f"字幕对齐音频不存在：{audio_path}")
    try:
        info = sf.info(str(audio_path))
    except RuntimeError as error:
        raise ValueError(f"无法读取字幕对齐音频时长：{audio_path}") from error
    samplerate = int(info.samplerate)
    frames = int(info.frames)
    if samplerate <= 0 or frames < 0:
        raise ValueError(f"字幕对齐音频时长无效：{audio_path}")
    duration_seconds = frames / samplerate
    frame_index = math.ceil(duration_seconds * DEFAULT_TIMELINE_FPS - 1e-9)
    return round(frame_index * 1000 / DEFAULT_TIMELINE_FPS)


def format_srt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError(f"SRT 时间戳不能为负数：{milliseconds}")

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
