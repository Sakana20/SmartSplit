from funasr_timeline.asr.base import AudioInfo
from funasr_timeline.forced_alignment import (
    ForcedAlignmentInfo,
    ForcedAlignmentResult,
    ForcedAlignmentUnit,
    map_forced_alignment_to_sentence_items,
)
from funasr_timeline.normalization import normalize_text
from funasr_timeline.segmentation import RegexSentenceSegmenter, attach_normalized_ranges


def test_forced_alignment_maps_units_to_sentence_ranges_with_punctuation_skipped() -> None:
    manuscript = "你好。English 123！"
    result = _forced_result(
        manuscript,
        [
            ForcedAlignmentUnit(0, "你", 10, 20, "你"),
            ForcedAlignmentUnit(1, "好", 20, 30, "好"),
            ForcedAlignmentUnit(2, "English", 50, 120, "english"),
            ForcedAlignmentUnit(3, "123", 120, 160, "123"),
        ],
    )
    segments = _segments(manuscript)

    items, timings = map_forced_alignment_to_sentence_items(segments, result, "qwen3-forced")

    assert [item.text for item in items] == ["你好。", "English 123！"]
    assert items[0].start_ms == 10
    assert items[0].end_ms == 30
    assert items[1].start_ms == 50
    assert items[1].end_ms == 160
    assert timings[1].unit_range == (2, 3)
    assert items[1].diagnostics["forced_unit_indexes"] == [2, 3]


def test_forced_alignment_records_text_mismatch_diagnostics() -> None:
    manuscript = "你好。"
    result = _forced_result(
        manuscript,
        [
            ForcedAlignmentUnit(0, "你", 10, 20, "你"),
        ],
    )
    segments = _segments(manuscript)

    items, timings = map_forced_alignment_to_sentence_items(segments, result, "qwen3-forced")

    assert items[0].status == "forced_missing_unit"
    assert timings[0].diagnostics["forced_text_mismatch"] is True
    assert timings[0].diagnostics["text_similarity"] < 1


def _segments(manuscript: str):
    normalized = normalize_text(manuscript)
    result = RegexSentenceSegmenter().segment(manuscript)
    return attach_normalized_ranges(result.segments, normalized)


def _forced_result(
    manuscript: str,
    units: list[ForcedAlignmentUnit],
) -> ForcedAlignmentResult:
    normalized_text = normalize_text(manuscript).text
    forced_normalized_text = "".join(unit.normalized_text for unit in units)
    return ForcedAlignmentResult(
        audio=AudioInfo("fixture.mp3", "mp3", 1000),
        aligner=ForcedAlignmentInfo("mock-forced", "fixture", "mps", "bfloat16", "Chinese"),
        input_text=manuscript,
        normalized_text=normalized_text,
        forced_normalized_text=forced_normalized_text,
        normalized_text_match=normalized_text == forced_normalized_text,
        units=units,
        diagnostics={},
    )
