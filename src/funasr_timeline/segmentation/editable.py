from __future__ import annotations

from pathlib import Path

from loguru import logger

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegment


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
    logger.debug("读取可编辑分句文件：{}", path)
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

    logger.debug(
        "可编辑分句读取完成：segments={} paragraphs={}", len(segments), paragraph_index + 1
    )
    return SegmentationResult(text="".join(text_parts), segments=segments)
