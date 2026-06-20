from __future__ import annotations

from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.render.base import TimelineRenderer


class SrtTimelineRenderer(TimelineRenderer):
    name = "srt"
    file_extension = ".srt"

    def render(self, items: list[SentenceTimelineItem]) -> str:
        blocks: list[str] = []
        cue_index = 1

        for item in items:
            if item.start_ms is None or item.end_ms is None:
                continue
            if item.end_ms < item.start_ms:
                continue

            blocks.append(
                "\n".join(
                    [
                        str(cue_index),
                        f"{format_srt_timestamp(item.start_ms)} --> "
                        f"{format_srt_timestamp(item.end_ms)}",
                        item.text,
                    ]
                )
            )
            cue_index += 1

        if not blocks:
            return ""
        return "\n\n".join(blocks) + "\n"


def format_srt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError(f"SRT 时间戳不能为负数：{milliseconds}")

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
