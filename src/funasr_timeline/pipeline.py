from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

from loguru import logger

from funasr_timeline.alignment import AlignmentResult, align_texts, asr_chars_as_dicts
from funasr_timeline.asr.base import AsrInfo, AsrService, AudioInfo, WordTimeline
from funasr_timeline.forced_alignment.base import ForcedAlignmentResult, ForcedAlignmentService
from funasr_timeline.forced_alignment.config import TelemetryConfig, TimelineProvider
from funasr_timeline.forced_alignment.sentence_mapper import (
    ForcedSentenceTiming,
    map_forced_alignment_to_sentence_items,
)
from funasr_timeline.manuscript import read_txt_manuscript
from funasr_timeline.merge import SentenceTimelineItem, merge_sentence_timelines
from funasr_timeline.normalization import NormalizedText, normalize_text
from funasr_timeline.render.srt import SrtTimelineRenderer
from funasr_timeline.report import build_alignment_report
from funasr_timeline.segmentation.base import SentenceSegment, SentenceSegmenter
from funasr_timeline.segmentation.editable import export_editable_segments, load_editable_segments
from funasr_timeline.segmentation.factory import segment_manuscript_text
from funasr_timeline.segmentation.normalization import attach_normalized_ranges
from funasr_timeline.segmentation.regex import RegexSentenceSegmenter
from funasr_timeline.sentence_matching import match_sentences_to_tokens


class _DiagnosticsAwareSegmenter(Protocol):
    def with_diagnostics_path(self, diagnostics_path: Path) -> SentenceSegmenter: ...


def run_pipeline(
    manuscript_path: Path,
    audio_path: Path,
    output_dir: Path,
    asr_service: AsrService | None = None,
    segmenter: SentenceSegmenter | None = None,
    segments_path: Path | None = None,
    timeline_provider: TimelineProvider = "asr-fuzzy",
    forced_alignment_service: ForcedAlignmentService | None = None,
    forced_alignment_language: str = "Chinese",
    telemetry_config: TelemetryConfig | None = None,
) -> dict[str, Path]:
    logger.debug(
        "开始完整流程：manuscript={} audio={} output_dir={}",
        manuscript_path,
        audio_path,
        output_dir,
    )
    _validate_audio(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manuscript = read_txt_manuscript(manuscript_path)
    logger.debug("稿件读取完成：chars={} path={}", len(manuscript.text), manuscript.path)
    if timeline_provider not in {"asr-fuzzy", "qwen3-forced", "hybrid"}:
        raise ValueError(f"不支持的 timeline_provider：{timeline_provider}")
    if timeline_provider in {"asr-fuzzy", "hybrid"} and asr_service is None:
        raise ValueError(f"{timeline_provider} 模式需要 asr_service")
    if timeline_provider in {"qwen3-forced", "hybrid"} and forced_alignment_service is None:
        raise ValueError(f"{timeline_provider} 模式需要 forced_alignment_service")

    segmenter = _with_segmentation_diagnostics(
        segmenter or RegexSentenceSegmenter(),
        output_dir / "llm_segmentation_diagnostics.json",
    )
    segmentation = (
        load_editable_segments(segments_path)
        if segments_path is not None
        else segment_manuscript_text(manuscript.text, segmenter)
    )
    logger.debug(
        "分句完成：strategy={} segments={} prepared_chars={}",
        f"editable:{segments_path}" if segments_path is not None else segmenter.name,
        len(segmentation.segments),
        len(segmentation.text),
    )
    normalized_manuscript = normalize_text(segmentation.text)
    segments = attach_normalized_ranges(segmentation.segments, normalized_manuscript)
    logger.debug(
        "稿件归一化完成：normalized_chars={} source_chars={}",
        len(normalized_manuscript.text),
        len(segmentation.text),
    )
    if timeline_provider in {"asr-fuzzy", "hybrid"}:
        assert asr_service is not None
        asr_branch = _run_asr_fuzzy_branch(
            audio_path=audio_path,
            asr_service=asr_service,
            normalized_manuscript=normalized_manuscript,
            segments=segments,
        )
    else:
        asr_branch = _empty_asr_branch(audio_path, normalized_manuscript)
    forced_result: ForcedAlignmentResult | None = None
    forced_timings: list[ForcedSentenceTiming] = []

    if timeline_provider in {"qwen3-forced", "hybrid"}:
        assert forced_alignment_service is not None
        forced_result = forced_alignment_service.align(
            audio_path=audio_path,
            text=segmentation.text,
            language=forced_alignment_language,
        )
        sentence_items, forced_timings = map_forced_alignment_to_sentence_items(
            segments=segments,
            forced_result=forced_result,
            timeline_provider=timeline_provider,
        )
        if timeline_provider == "hybrid":
            sentence_items = _attach_asr_fuzzy_diagnostics(
                forced_items=sentence_items,
                asr_items=asr_branch.sentence_items,
            )
        logger.debug(
            "forced alignment 句子时间轴完成：items={} provider={}",
            len(sentence_items),
            timeline_provider,
        )
    else:
        sentence_items = asr_branch.sentence_items
        logger.debug(
            "ASR fuzzy 句子时间轴合并完成：items={} low_confidence={} no_match={}",
            len(sentence_items),
            sum(1 for item in sentence_items if item.status == "low_confidence"),
            sum(1 for item in sentence_items if item.status == "no_match"),
        )
    srt_renderer = SrtTimelineRenderer()
    telemetry = _build_telemetry(
        timeline_provider=timeline_provider,
        primary="qwen3-forced" if timeline_provider in {"qwen3-forced", "hybrid"} else "asr-fuzzy",
        forced_result=forced_result,
        forced_timings=forced_timings,
        asr_branch=asr_branch,
        final_items=sentence_items,
        telemetry_config=telemetry_config or TelemetryConfig(),
    )

    report = build_alignment_report(
        manuscript_path=manuscript.path,
        word_timeline=asr_branch.word_timeline,
        normalized_manuscript=normalized_manuscript,
        segments=segments,
        sentence_items=sentence_items,
        alignment=asr_branch.alignment,
        segmenter_name=f"editable:{segments_path}" if segments_path is not None else segmenter.name,
        telemetry_summary=_telemetry_summary(telemetry),
    )

    paths = {
        "word_timeline": output_dir / "word_timeline.json",
        "manuscript_segments": output_dir / "manuscript_segments.json",
        "normalized_text": output_dir / "normalized_text.json",
        "alignment": output_dir / "alignment.json",
        "sentence_timeline": output_dir / "sentence_timeline.json",
        "sentence_timeline_srt": output_dir / f"sentence_timeline{srt_renderer.file_extension}",
        "alignment_report": output_dir / "alignment_report.json",
    }
    if forced_result is not None:
        paths["forced_alignment"] = output_dir / "forced_alignment.json"
        paths["telemetry"] = output_dir / "telemetry.json"

    _write_json(paths["word_timeline"], asr_branch.word_timeline.to_dict())
    _write_json(paths["manuscript_segments"], [segment.to_dict() for segment in segments])
    _write_json(
        paths["normalized_text"],
        {
            "manuscript": {
                "text": normalized_manuscript.text,
                "chars": [
                    {
                        "normalized_index": char.normalized_index,
                        "original_index": char.original_index,
                        "original_char": char.original_char,
                        "normalized_char": char.normalized_char,
                    }
                    for char in normalized_manuscript.chars
                ],
            },
            "asr": {
                "text": asr_branch.alignment.asr_text,
                "chars": asr_chars_as_dicts(asr_branch.alignment.asr_chars),
            },
        },
    )
    _write_json(
        paths["alignment"],
        {
            "global_match_score": round(asr_branch.alignment.global_match_score, 6),
            "manuscript_to_token": {
                str(key): value for key, value in asr_branch.alignment.manuscript_to_token.items()
            },
            "opcodes": asr_branch.alignment.opcodes_as_dicts(),
            "unmatched_manuscript_indexes": asr_branch.alignment.unmatched_manuscript_indexes,
            "unmapped_asr_indexes": asr_branch.alignment.unmapped_asr_indexes,
        },
    )
    _write_json(paths["sentence_timeline"], [item.to_dict() for item in sentence_items])
    _write_text(paths["sentence_timeline_srt"], srt_renderer.render(sentence_items))
    _write_json(paths["alignment_report"], report)
    if forced_result is not None:
        _write_json(paths["forced_alignment"], forced_result.to_dict())
        _write_json(paths["telemetry"], telemetry)
    logger.debug("完整流程输出完成：output_dir={}", output_dir)

    return paths


class _AsrFuzzyBranch:
    def __init__(
        self,
        word_timeline: WordTimeline,
        alignment: AlignmentResult,
        sentence_items: list[SentenceTimelineItem],
    ) -> None:
        self.word_timeline = word_timeline
        self.alignment = alignment
        self.sentence_items = sentence_items


def _run_asr_fuzzy_branch(
    audio_path: Path,
    asr_service: AsrService,
    normalized_manuscript: NormalizedText,
    segments: list[SentenceSegment],
) -> _AsrFuzzyBranch:
    word_timeline = asr_service.transcribe(audio_path)
    logger.debug(
        "ASR 时间轴生成完成：provider={} tokens={} duration_ms={}",
        word_timeline.asr.provider,
        len(word_timeline.tokens),
        word_timeline.audio.duration_ms,
    )
    alignment = align_texts(normalized_manuscript.text, word_timeline.tokens)
    logger.debug(
        "全文对齐完成：score={:.4f} unmatched_manuscript={} unmapped_asr={}",
        alignment.global_match_score,
        len(alignment.unmatched_manuscript_indexes),
        len(alignment.unmapped_asr_indexes),
    )
    sentence_matches = match_sentences_to_tokens(segments, word_timeline.tokens)
    sentence_items = merge_sentence_timelines(segments, word_timeline.tokens, sentence_matches)
    return _AsrFuzzyBranch(
        word_timeline=word_timeline,
        alignment=alignment,
        sentence_items=sentence_items,
    )


def _empty_asr_branch(
    audio_path: Path,
    normalized_manuscript: NormalizedText,
) -> _AsrFuzzyBranch:
    word_timeline = WordTimeline(
        audio=AudioInfo(
            path=str(audio_path),
            format=audio_path.suffix.lstrip(".").lower(),
            duration_ms=None,
        ),
        asr=AsrInfo(provider="none", model=None, text=""),
        tokens=[],
    )
    return _AsrFuzzyBranch(
        word_timeline=word_timeline,
        alignment=align_texts(normalized_manuscript.text, []),
        sentence_items=[],
    )


def _attach_asr_fuzzy_diagnostics(
    forced_items: list[SentenceTimelineItem],
    asr_items: list[SentenceTimelineItem],
) -> list[SentenceTimelineItem]:
    asr_by_index = {item.index: item for item in asr_items}
    items: list[SentenceTimelineItem] = []
    for forced_item in forced_items:
        asr_item = asr_by_index.get(forced_item.index)
        diagnostics = dict(forced_item.diagnostics)
        if asr_item is not None:
            diagnostics["asr_fuzzy"] = _sentence_item_summary(asr_item)
        items.append(
            SentenceTimelineItem(
                index=forced_item.index,
                text=forced_item.text,
                paragraph_index=forced_item.paragraph_index,
                start_ms=forced_item.start_ms,
                end_ms=forced_item.end_ms,
                duration_ms=forced_item.duration_ms,
                raw_start_ms=forced_item.raw_start_ms,
                raw_end_ms=forced_item.raw_end_ms,
                time_adjusted=forced_item.time_adjusted,
                match_score=forced_item.match_score,
                status=forced_item.status,
                matched_token_indexes=asr_item.matched_token_indexes if asr_item else [],
                matched_asr_text=asr_item.matched_asr_text if asr_item else "",
                normalized_text=forced_item.normalized_text,
                manuscript_char_range=forced_item.manuscript_char_range,
                normalized_char_range=forced_item.normalized_char_range,
                asr_token_range=asr_item.asr_token_range if asr_item else (None, None),
                diagnostics=diagnostics,
            )
        )
    return items


def _build_telemetry(
    timeline_provider: str,
    primary: str,
    forced_result: ForcedAlignmentResult | None,
    forced_timings: list[ForcedSentenceTiming],
    asr_branch: _AsrFuzzyBranch,
    final_items: list[SentenceTimelineItem],
    telemetry_config: TelemetryConfig,
) -> dict[str, Any]:
    forced_timing_by_index = {timing.sentence_index: timing for timing in forced_timings}
    asr_item_by_index = {item.index: item for item in asr_branch.sentence_items}
    sentences: list[dict[str, Any]] = []
    for item in final_items:
        forced_timing = forced_timing_by_index.get(item.index)
        asr_item = asr_item_by_index.get(item.index)
        payload: dict[str, Any] = {"index": item.index}
        if forced_timing is not None:
            payload["forced"] = forced_timing.to_telemetry()
        if asr_item is not None:
            payload["asr_fuzzy"] = _sentence_item_summary(asr_item)
        if (
            telemetry_config.include_sentence_comparison
            and forced_timing is not None
            and asr_item is not None
        ):
            payload["comparison"] = _compare_sentence_timing(forced_timing, asr_item)
        sentences.append(payload)

    telemetry: dict[str, Any] = {
        "timeline_provider": timeline_provider,
        "primary": primary,
        "asr_fuzzy": {
            "provider": asr_branch.word_timeline.asr.provider,
            "model": asr_branch.word_timeline.asr.model,
            "token_count": len(asr_branch.word_timeline.tokens),
            "global_match_score": round(asr_branch.alignment.global_match_score, 6),
        },
        "sentences": sentences,
    }
    if telemetry_config.include_asr_tokens:
        telemetry["asr_fuzzy"]["tokens"] = [
            {
                "index": token.index,
                "text": token.text,
                "start_ms": token.start_ms,
                "end_ms": token.end_ms,
                "confidence": token.confidence,
                "source": token.source,
            }
            for token in asr_branch.word_timeline.tokens
        ]
    if forced_result is not None:
        forced_payload: dict[str, Any] = {
            "provider": forced_result.aligner.provider,
            "model": forced_result.aligner.model,
            "device_map": forced_result.aligner.device_map,
            "dtype": forced_result.aligner.dtype,
            "language": forced_result.aligner.language,
            "unit_count": len(forced_result.units),
            "normalized_text_match": forced_result.normalized_text_match,
        }
        if telemetry_config.include_forced_units:
            forced_payload["units"] = [
                {
                    "index": unit.index,
                    "text": unit.text,
                    "normalized_text": unit.normalized_text,
                    "start_ms": unit.start_ms,
                    "end_ms": unit.end_ms,
                }
                for unit in forced_result.units
            ]
        telemetry["forced_alignment"] = forced_payload
    return telemetry


def _sentence_item_summary(item: SentenceTimelineItem) -> dict[str, Any]:
    return {
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "duration_ms": item.duration_ms,
        "status": item.status,
        "match_score": item.match_score,
        "token_range": list(item.asr_token_range),
        "matched_token_indexes": item.matched_token_indexes,
        "matched_asr_text": item.matched_asr_text,
    }


def _compare_sentence_timing(
    forced_timing: ForcedSentenceTiming,
    asr_item: SentenceTimelineItem,
) -> dict[str, int | None]:
    return {
        "start_delta_ms": _delta(forced_timing.start_ms, asr_item.start_ms),
        "end_delta_ms": _delta(forced_timing.end_ms, asr_item.end_ms),
        "duration_delta_ms": _delta(forced_timing.duration_ms, asr_item.duration_ms),
    }


def _delta(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left - right


def _telemetry_summary(telemetry: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "timeline_provider": telemetry["timeline_provider"],
        "primary": telemetry["primary"],
        "asr_fuzzy": {
            key: value
            for key, value in telemetry["asr_fuzzy"].items()
            if key in {"provider", "model", "token_count", "global_match_score"}
        },
    }
    if "forced_alignment" in telemetry:
        summary["forced_alignment"] = {
            key: value
            for key, value in telemetry["forced_alignment"].items()
            if key
            in {
                "provider",
                "model",
                "device_map",
                "dtype",
                "language",
                "unit_count",
                "normalized_text_match",
            }
        }
    return summary


def run_segmentation(
    manuscript_path: Path,
    output_dir: Path,
    segmenter: SentenceSegmenter,
) -> dict[str, Path]:
    logger.debug(
        "开始独立分句：manuscript={} output_dir={} segmenter={}",
        manuscript_path,
        output_dir,
        segmenter.name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript = read_txt_manuscript(manuscript_path)
    segmenter = _with_segmentation_diagnostics(
        segmenter,
        output_dir / "llm_segmentation_diagnostics.json",
    )
    segmentation = segment_manuscript_text(manuscript.text, segmenter)
    normalized_manuscript = normalize_text(segmentation.text)
    segments = attach_normalized_ranges(segmentation.segments, normalized_manuscript)
    logger.debug(
        "独立分句完成：segments={} normalized_chars={}",
        len(segments),
        len(normalized_manuscript.text),
    )

    paths = {
        "editable_segments": output_dir / "editable_segments.txt",
        "manuscript_segments": output_dir / "manuscript_segments.json",
    }
    _write_text(paths["editable_segments"], export_editable_segments(segments))
    _write_json(paths["manuscript_segments"], [segment.to_dict() for segment in segments])
    logger.debug("独立分句输出完成：output_dir={}", output_dir)
    return paths


def _validate_audio(audio_path: Path) -> None:
    if audio_path.suffix.lower() != ".mp3":
        raise ValueError(f"第一阶段仅支持 .mp3 音频：{audio_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")


def _with_segmentation_diagnostics(
    segmenter: SentenceSegmenter,
    diagnostics_path: Path,
) -> SentenceSegmenter:
    if hasattr(segmenter, "with_diagnostics_path"):
        return cast(_DiagnosticsAwareSegmenter, segmenter).with_diagnostics_path(diagnostics_path)
    return segmenter


def _write_json(path: Path, payload: Any) -> None:
    logger.debug("写入 JSON：{}", path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    logger.debug("写入文本：{} chars={}", path, len(payload))
    path.write_text(payload, encoding="utf-8")
