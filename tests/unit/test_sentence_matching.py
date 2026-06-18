from funasr_timeline.asr.base import AsrToken
from funasr_timeline.merge import merge_sentence_timelines
from funasr_timeline.normalization import normalize_text
from funasr_timeline.segmentation import RegexSentenceSegmenter, attach_normalized_ranges
from funasr_timeline.sentence_matching import match_sentences_to_tokens


def _segments(manuscript: str):
    normalized = normalize_text(manuscript)
    result = RegexSentenceSegmenter().segment(manuscript)
    return attach_normalized_ranges(result.segments, normalized)


def test_fuzzy_matching_skips_extra_asr_prefix_in_order() -> None:
    segments = _segments("你好。再见。")
    tokens = [
        AsrToken(index=0, text="嗯", start_ms=0, end_ms=10),
        AsrToken(index=1, text="你", start_ms=10, end_ms=20),
        AsrToken(index=2, text="好", start_ms=20, end_ms=30),
        AsrToken(index=3, text="再", start_ms=40, end_ms=50),
        AsrToken(index=4, text="见", start_ms=50, end_ms=60),
    ]

    matches = match_sentences_to_tokens(segments, tokens)

    assert matches[0].status == "ok"
    assert matches[0].matched_token_indexes == [1, 2]
    assert matches[0].matched_asr_text == "你好"
    assert matches[1].matched_token_indexes == [3, 4]


def test_fuzzy_matching_marks_low_confidence_but_keeps_diagnostics() -> None:
    segments = _segments("你好。")
    tokens = [
        AsrToken(index=0, text="你", start_ms=10, end_ms=20),
        AsrToken(index=1, text="坏", start_ms=20, end_ms=30),
    ]

    matches = match_sentences_to_tokens(segments, tokens, low_confidence_threshold=0.9)

    assert matches[0].status == "low_confidence"
    assert matches[0].match_score < 0.9
    assert matches[0].diagnostics["unmatched_manuscript_chars"] == [
        {"normalized_index": 1, "char": "好"}
    ]


def test_fuzzy_matching_uses_selected_asr_window_for_timeline() -> None:
    segments = _segments("12元红包。")
    tokens = [
        AsrToken(index=0, text="十", start_ms=100, end_ms=200),
        AsrToken(index=1, text="二", start_ms=200, end_ms=300),
        AsrToken(index=2, text="元", start_ms=300, end_ms=400),
        AsrToken(index=3, text="红", start_ms=400, end_ms=500),
        AsrToken(index=4, text="包", start_ms=500, end_ms=600),
    ]

    matches = match_sentences_to_tokens(segments, tokens)
    items = merge_sentence_timelines(segments, tokens, matches)

    assert matches[0].matched_asr_text == "十二元红包"
    assert matches[0].matched_token_indexes == [0, 1, 2, 3, 4]
    assert matches[0].diagnostics["exact_matched_token_indexes"] == [0, 1, 2, 3, 4]
    assert items[0].raw_start_ms == 100
    assert items[0].raw_end_ms == 600


def test_merge_adjusts_overlapping_sentence_times() -> None:
    segments = _segments("你好。再见。")
    tokens = [
        AsrToken(index=0, text="你", start_ms=10, end_ms=60),
        AsrToken(index=1, text="好", start_ms=60, end_ms=100),
        AsrToken(index=2, text="再", start_ms=90, end_ms=120),
        AsrToken(index=3, text="见", start_ms=120, end_ms=150),
    ]
    matches = match_sentences_to_tokens(segments, tokens)

    items = merge_sentence_timelines(segments, tokens, matches)

    assert items[0].start_ms == 10
    assert items[0].end_ms == 100
    assert items[1].raw_start_ms == 90
    assert items[1].start_ms == 100
    assert items[1].end_ms == 150
    assert items[1].time_adjusted is True
    assert items[1].status == "ok"
