from funasr_timeline.alignment import align_texts
from funasr_timeline.asr.base import AsrToken
from funasr_timeline.merge import merge_sentence_timelines
from funasr_timeline.normalization import normalize_text
from funasr_timeline.segmentation import RegexSentenceSegmenter, attach_normalized_ranges
from funasr_timeline.sentence_matching import match_sentences_to_tokens


def test_alignment_maps_equal_chars_after_extra_asr_prefix() -> None:
    tokens = [
        AsrToken(index=0, text="嗯", start_ms=0, end_ms=50),
        AsrToken(index=1, text="你", start_ms=50, end_ms=100),
        AsrToken(index=2, text="好", start_ms=100, end_ms=150),
    ]

    alignment = align_texts("你好", tokens)

    assert alignment.manuscript_to_token == {0: 1, 1: 2}
    assert alignment.unmapped_asr_indexes == [0]
    assert alignment.global_match_score == 1.0


def test_merge_sentence_timelines_uses_manuscript_text_and_token_times() -> None:
    manuscript = "你好。再见。"
    normalized = normalize_text(manuscript)
    result = RegexSentenceSegmenter().segment(manuscript)
    segments = attach_normalized_ranges(result.segments, normalized)
    tokens = [
        AsrToken(index=0, text="你", start_ms=10, end_ms=20),
        AsrToken(index=1, text="好", start_ms=20, end_ms=30),
        AsrToken(index=2, text="再", start_ms=40, end_ms=50),
        AsrToken(index=3, text="见", start_ms=50, end_ms=60),
    ]
    matches = match_sentences_to_tokens(segments, tokens)

    items = merge_sentence_timelines(segments, tokens, matches)

    assert [item.text for item in items] == ["你好。", "再见。"]
    assert items[0].start_ms == 10
    assert items[0].end_ms == 30
    assert items[0].status == "ok"
    assert items[1].matched_token_indexes == [2, 3]
