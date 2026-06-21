from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegmenter
from funasr_timeline.segmentation.hanlp import HanlpSegmenter
from funasr_timeline.segmentation.jieba_subtitle import JiebaSubtitleSegmenter
from funasr_timeline.segmentation.regex import RegexSentenceSegmenter

SEGMENTER_FACTORIES: dict[str, Callable[[], SentenceSegmenter]] = {
    RegexSentenceSegmenter.name: RegexSentenceSegmenter,
    HanlpSegmenter.name: HanlpSegmenter,
    JiebaSubtitleSegmenter.name: JiebaSubtitleSegmenter,
}
LLM_SEGMENTER_NAME = "llm"


def available_segmenters() -> tuple[str, ...]:
    return tuple(sorted((*SEGMENTER_FACTORIES, LLM_SEGMENTER_NAME)))


def available_llm_fallback_segmenters() -> tuple[str, ...]:
    return tuple(sorted(SEGMENTER_FACTORIES))


def create_segmenter(
    name: str,
    llm_config_path: Path | None = None,
    segment_threshold: int = 10,
    llm_fallback_segmenter: str = HanlpSegmenter.name,
    llm_raise_on_error: bool = False,
) -> SentenceSegmenter:
    if name == HanlpSegmenter.name:
        return HanlpSegmenter(threshold=segment_threshold)
    if name == LLM_SEGMENTER_NAME:
        from funasr_timeline.segmentation.llm import (
            DEFAULT_LLM_CONFIG_PATH,
            LlmSentenceSegmenter,
            load_llm_segmentation_config,
        )

        fallback = create_segmenter(
            llm_fallback_segmenter,
            segment_threshold=segment_threshold,
        )
        return LlmSentenceSegmenter(
            load_llm_segmentation_config(llm_config_path or DEFAULT_LLM_CONFIG_PATH),
            fallback_segmenter=fallback,
            raise_on_error=llm_raise_on_error,
        )

    try:
        return SEGMENTER_FACTORIES[name]()
    except KeyError as error:
        available = "、".join(available_segmenters())
        raise ValueError(f"未知分句实现：{name}。可选值：{available}") from error


def segment_manuscript_text(text: str, segmenter: SentenceSegmenter) -> SegmentationResult:
    return segmenter.segment(text)
