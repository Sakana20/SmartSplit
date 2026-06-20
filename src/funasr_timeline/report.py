from __future__ import annotations

from typing import Any

from funasr_timeline.alignment import AlignmentResult
from funasr_timeline.asr.base import WordTimeline
from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.normalization import NormalizedText
from funasr_timeline.segmentation.base import SentenceSegment


def build_alignment_report(
    manuscript_path: str,
    word_timeline: WordTimeline,
    normalized_manuscript: NormalizedText,
    segments: list[SentenceSegment],
    sentence_items: list[SentenceTimelineItem],
    alignment: AlignmentResult,
    segmenter_name: str,
    telemetry_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_by_index = {token.index: token for token in word_timeline.tokens}
    low_confidence = [
        item.to_dict()
        for item in sentence_items
        if item.status
        in {
            "low_confidence",
            "no_match",
            "empty_after_normalization",
            "invalid_time_range",
        }
    ]

    report = {
        "inputs": {
            "manuscript": manuscript_path,
            "audio": word_timeline.audio.path,
            "audio_format": word_timeline.audio.format,
        },
        "asr": {
            "provider": word_timeline.asr.provider,
            "model": word_timeline.asr.model,
            "text": word_timeline.asr.text,
            "token_count": len(word_timeline.tokens),
        },
        "normalization": {
            "strategy": "nfkc_lower_remove_punctuation_symbols_whitespace",
            "manuscript_normalized_text": normalized_manuscript.text,
            "asr_normalized_text": alignment.asr_text,
        },
        "segmentation": {
            "strategy": segmenter_name,
            "sentence_count": len(segments),
            "sentences": [segment.to_dict() for segment in segments],
        },
        "alignment": {
            "global_match_score": round(alignment.global_match_score, 6),
            "opcodes": alignment.opcodes_as_dicts(),
            "unmatched_manuscript_chars": [
                {
                    "normalized_index": index,
                    "char": alignment.manuscript_text[index],
                    "original_index": normalized_manuscript.chars[index].original_index,
                    "original_char": normalized_manuscript.chars[index].original_char,
                }
                for index in alignment.unmatched_manuscript_indexes
            ],
            "unmapped_asr_tokens": [
                {
                    "normalized_index": index,
                    "token_index": alignment.asr_chars[index].token_index,
                    "token_text": alignment.asr_chars[index].token_text,
                    "normalized_char": alignment.asr_chars[index].normalized_char,
                    "start_ms": token_by_index[alignment.asr_chars[index].token_index].start_ms,
                    "end_ms": token_by_index[alignment.asr_chars[index].token_index].end_ms,
                }
                for index in alignment.unmapped_asr_indexes
            ],
        },
        "low_confidence_sentences": low_confidence,
    }
    if telemetry_summary is not None:
        report["telemetry"] = telemetry_summary
    return report
