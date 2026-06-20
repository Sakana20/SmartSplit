from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegmenter
from funasr_timeline.segmentation.jieba_subtitle import JiebaSubtitleSegmenter
from funasr_timeline.segmentation.regex import RegexSentenceSegmenter

SEGMENTER_FACTORIES: dict[str, Callable[[], SentenceSegmenter]] = {
    RegexSentenceSegmenter.name: RegexSentenceSegmenter,
    JiebaSubtitleSegmenter.name: JiebaSubtitleSegmenter,
}
LLM_SEGMENTER_NAME = "llm"


def available_segmenters() -> tuple[str, ...]:
    return tuple(sorted((*SEGMENTER_FACTORIES, LLM_SEGMENTER_NAME)))


def create_segmenter(name: str, llm_config_path: Path | None = None) -> SentenceSegmenter:
    if name == LLM_SEGMENTER_NAME:
        from funasr_timeline.segmentation.llm import (
            DEFAULT_LLM_CONFIG_PATH,
            LlmSentenceSegmenter,
            load_llm_segmentation_config,
        )

        return LlmSentenceSegmenter(
            load_llm_segmentation_config(llm_config_path or DEFAULT_LLM_CONFIG_PATH)
        )

    try:
        return SEGMENTER_FACTORIES[name]()
    except KeyError as error:
        available = "、".join(available_segmenters())
        raise ValueError(f"未知分句实现：{name}。可选值：{available}") from error


def segment_manuscript_text(text: str, segmenter: SentenceSegmenter) -> SegmentationResult:
    return segmenter.segment(text)
