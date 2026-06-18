from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from funasr_timeline.alignment import align_texts, asr_chars_as_dicts
from funasr_timeline.asr.base import AsrService
from funasr_timeline.manuscript import read_txt_manuscript
from funasr_timeline.merge import merge_sentence_timelines
from funasr_timeline.normalization import normalize_text
from funasr_timeline.render import SrtTimelineRenderer
from funasr_timeline.report import build_alignment_report
from funasr_timeline.segmentation import (
    RegexSentenceSegmenter,
    SentenceSegmenter,
    attach_normalized_ranges,
    export_editable_segments,
    load_editable_segments,
    segment_manuscript_text,
)
from funasr_timeline.sentence_matching import match_sentences_to_tokens


def run_pipeline(
    manuscript_path: Path,
    audio_path: Path,
    output_dir: Path,
    asr_service: AsrService,
    segmenter: SentenceSegmenter | None = None,
    segments_path: Path | None = None,
) -> dict[str, Path]:
    _validate_audio(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manuscript = read_txt_manuscript(manuscript_path)
    word_timeline = asr_service.transcribe(audio_path)

    segmenter = segmenter or RegexSentenceSegmenter()
    segmentation = (
        load_editable_segments(segments_path)
        if segments_path is not None
        else segment_manuscript_text(manuscript.text, segmenter)
    )
    normalized_manuscript = normalize_text(segmentation.text)
    segments = attach_normalized_ranges(segmentation.segments, normalized_manuscript)
    alignment = align_texts(normalized_manuscript.text, word_timeline.tokens)
    sentence_matches = match_sentences_to_tokens(segments, word_timeline.tokens)
    sentence_items = merge_sentence_timelines(segments, word_timeline.tokens, sentence_matches)
    srt_renderer = SrtTimelineRenderer()

    report = build_alignment_report(
        manuscript_path=manuscript.path,
        word_timeline=word_timeline,
        normalized_manuscript=normalized_manuscript,
        segments=segments,
        sentence_items=sentence_items,
        alignment=alignment,
        segmenter_name=f"editable:{segments_path}" if segments_path is not None else segmenter.name,
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

    _write_json(paths["word_timeline"], word_timeline.to_dict())
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
                "text": alignment.asr_text,
                "chars": asr_chars_as_dicts(alignment.asr_chars),
            },
        },
    )
    _write_json(
        paths["alignment"],
        {
            "global_match_score": round(alignment.global_match_score, 6),
            "manuscript_to_token": {
                str(key): value for key, value in alignment.manuscript_to_token.items()
            },
            "opcodes": alignment.opcodes_as_dicts(),
            "unmatched_manuscript_indexes": alignment.unmatched_manuscript_indexes,
            "unmapped_asr_indexes": alignment.unmapped_asr_indexes,
        },
    )
    _write_json(paths["sentence_timeline"], [item.to_dict() for item in sentence_items])
    _write_text(paths["sentence_timeline_srt"], srt_renderer.render(sentence_items))
    _write_json(paths["alignment_report"], report)

    return paths


def run_segmentation(
    manuscript_path: Path,
    output_dir: Path,
    segmenter: SentenceSegmenter,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript = read_txt_manuscript(manuscript_path)
    segmentation = segment_manuscript_text(manuscript.text, segmenter)
    normalized_manuscript = normalize_text(segmentation.text)
    segments = attach_normalized_ranges(segmentation.segments, normalized_manuscript)

    paths = {
        "editable_segments": output_dir / "editable_segments.txt",
        "manuscript_segments": output_dir / "manuscript_segments.json",
    }
    _write_text(paths["editable_segments"], export_editable_segments(segments))
    _write_json(paths["manuscript_segments"], [segment.to_dict() for segment in segments])
    return paths


def _validate_audio(audio_path: Path) -> None:
    if audio_path.suffix.lower() != ".mp3":
        raise ValueError(f"第一阶段仅支持 .mp3 音频：{audio_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")
