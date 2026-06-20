from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from funasr_timeline.segmentation.base import SentenceSegment

NO_SPLIT_START = "[[NO_SPLIT]]"
NO_SPLIT_END = "[[/NO_SPLIT]]"


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    start: int
    end: int
    paragraph_index: int
    protected: bool


def append_protected_segment(segments: list[SentenceSegment], block: TextBlock) -> None:
    sentence_text = block.text.strip()
    if not sentence_text:
        return
    segments.append(
        SentenceSegment(
            index=len(segments),
            text=sentence_text,
            paragraph_index=block.paragraph_index,
            char_start=block.start + leading_whitespace_len(block.text),
            char_end=block.end - trailing_whitespace_len(block.text),
            boundary="protected",
        )
    )


def split_text_blocks(text: str) -> tuple[str, list[TextBlock]]:
    prepared_parts: list[str] = []
    raw_position = 0
    clean_position = 0
    paragraph_index = -1
    blocks: list[TextBlock] = []

    while raw_position < len(text):
        start_marker = text.find(NO_SPLIT_START, raw_position)
        if start_marker == -1:
            clean_position, paragraph_index = _append_unprotected_blocks(
                text[raw_position:],
                prepared_parts,
                blocks,
                clean_position,
                paragraph_index,
            )
            break

        clean_position, paragraph_index = _append_unprotected_blocks(
            text[raw_position:start_marker],
            prepared_parts,
            blocks,
            clean_position,
            paragraph_index,
        )
        protected_start = start_marker + len(NO_SPLIT_START)
        end_marker = text.find(NO_SPLIT_END, protected_start)
        if end_marker == -1:
            raise ValueError(f"缺少不分句结束标记：{NO_SPLIT_END}")

        protected_text = text[protected_start:end_marker]
        if protected_text:
            if paragraph_index < 0:
                paragraph_index = 0
            start = clean_position
            # NO_SPLIT 标记会从 prepared_text 中移除，但内部文本作为单独 block 保留。
            prepared_parts.append(protected_text)
            clean_position += len(protected_text)
            blocks.append(
                TextBlock(
                    text=protected_text,
                    start=start,
                    end=clean_position,
                    paragraph_index=paragraph_index,
                    protected=True,
                )
            )
        raw_position = end_marker + len(NO_SPLIT_END)

    protected_count = sum(1 for block in blocks if block.protected)
    logger.debug("文本块拆分完成：blocks={} protected={}", len(blocks), protected_count)
    return "".join(prepared_parts), blocks


def leading_whitespace_len(text: str) -> int:
    return len(text) - len(text.lstrip())


def trailing_whitespace_len(text: str) -> int:
    return len(text) - len(text.rstrip())


def _append_unprotected_blocks(
    text: str,
    prepared_parts: list[str],
    blocks: list[TextBlock],
    clean_position: int,
    paragraph_index: int,
) -> tuple[int, int]:
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        newline = line[len(content) :]
        line_start = clean_position
        continues_existing_paragraph = bool(prepared_parts) and not _last_part_ends_with_newline(
            prepared_parts
        )
        prepared_parts.append(content)
        clean_position += len(content)

        if content.strip():
            if not continues_existing_paragraph:
                paragraph_index += 1
            blocks.append(
                TextBlock(
                    text=content,
                    start=line_start,
                    end=clean_position,
                    paragraph_index=paragraph_index,
                    protected=False,
                )
            )

        if newline:
            prepared_parts.append(newline)
            clean_position += len(newline)

    return clean_position, paragraph_index


def _last_part_ends_with_newline(parts: list[str]) -> bool:
    for part in reversed(parts):
        if part:
            return part.endswith(("\n", "\r"))
    return False
