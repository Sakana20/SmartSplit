from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.render import SrtTimelineRenderer, format_srt_timestamp


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(3_725_006) == "01:02:05,006"


def test_srt_renderer_uses_sentence_text_and_skips_missing_times() -> None:
    items = [
        _item(index=0, text="第一句话。", start_ms=100, end_ms=2500),
        _item(index=1, text="无时间。", start_ms=None, end_ms=None),
        _item(index=2, text="第二句话。", start_ms=2600, end_ms=4000),
    ]

    rendered = SrtTimelineRenderer().render(items)

    assert rendered == (
        "1\n"
        "00:00:00,100 --> 00:00:02,500\n"
        "第一句话。\n"
        "\n"
        "2\n"
        "00:00:02,600 --> 00:00:04,000\n"
        "第二句话。\n"
    )


def _item(
    index: int,
    text: str,
    start_ms: int | None,
    end_ms: int | None,
) -> SentenceTimelineItem:
    duration_ms = end_ms - start_ms if start_ms is not None and end_ms is not None else None
    return SentenceTimelineItem(
        index=index,
        text=text,
        paragraph_index=0,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        raw_start_ms=start_ms,
        raw_end_ms=end_ms,
        time_adjusted=False,
        match_score=1.0,
        status="ok",
        matched_token_indexes=[],
        matched_asr_text="",
        normalized_text="",
        manuscript_char_range=(0, len(text)),
        normalized_char_range=(None, None),
        asr_token_range=(None, None),
        diagnostics={},
    )
