import wave
from pathlib import Path

from funasr_timeline.merge import SentenceTimelineItem
from funasr_timeline.render import (
    SrtTimelineRenderer,
    SubtitleCue,
    audio_duration_ms,
    format_srt_timestamp,
    postprocess_subtitle_cues,
)


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
        "00:00:00,000 --> 00:00:02,500\n"
        "第一句话。\n"
        "\n"
        "2\n"
        "00:00:02,600 --> 00:00:04,000\n"
        "第二句话。\n"
    )
    assert items[0].start_ms == 100


def test_srt_renderer_can_keep_first_cue_speech_start() -> None:
    items = [_item(index=0, text="第一句话。", start_ms=100, end_ms=2500)]

    rendered = SrtTimelineRenderer(align_first_subtitle_to_audio_start=False).render(items)

    assert "00:00:00,100 --> 00:00:02,500" in rendered


def test_srt_renderer_aligns_only_last_rendered_cue_to_audio_end(tmp_path: Path) -> None:
    audio_path = tmp_path / "tts.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(1000)
        audio.writeframes(b"\0\0" * 4501)
    items = [
        _item(index=0, text="第一句。", start_ms=100, end_ms=1000),
        _item(index=1, text="无时间。", start_ms=None, end_ms=None),
        _item(index=2, text="最后一句。", start_ms=1200, end_ms=3000),
    ]

    rendered = SrtTimelineRenderer(
        subtitle_alignment_audio=audio_path,
        align_first_subtitle_to_audio_start=False,
    ).render(items)

    assert audio_duration_ms(audio_path) == 4533
    assert "00:00:00,100 --> 00:00:01,000" in rendered
    assert "00:00:01,200 --> 00:00:04,533" in rendered
    assert items[-1].end_ms == 3000


def test_srt_renderer_rejects_audio_ending_before_last_cue(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(1000)
        audio.writeframes(b"\0\0" * 500)

    try:
        SrtTimelineRenderer(subtitle_alignment_audio=audio_path).render(
            [_item(index=0, text="字幕。", start_ms=600, end_ms=900)]
        )
    except ValueError as error:
        assert "早于最后一条字幕的开始时间" in str(error)
    else:
        raise AssertionError("expected invalid alignment audio duration to fail")


def test_postprocess_fills_only_gaps_within_threshold() -> None:
    result = postprocess_subtitle_cues(
        [
            SubtitleCue(source_index=0, text="第一句", start_ms=0, end_ms=500),
            SubtitleCue(source_index=1, text="第二句", start_ms=567, end_ms=1000),
            SubtitleCue(source_index=2, text="第三句", start_ms=1068, end_ms=1500),
        ],
        gap_threshold_ms=67,
        minimum_duration_ms=0,
    )

    assert result.cues[0].end_ms == 567
    assert result.cues[1].end_ms == 1000
    assert result.adjustments == [
        {
            "type": "short_gap_filled",
            "source_index": 0,
            "next_source_index": 1,
            "gap_ms": 67,
            "original_end_ms": 500,
            "adjusted_end_ms": 567,
        }
    ]


def test_postprocess_extends_short_cue_without_overlap() -> None:
    original = SubtitleCue(source_index=0, text="短句", start_ms=100, end_ms=150)
    result = postprocess_subtitle_cues(
        [
            original,
            SubtitleCue(source_index=1, text="下一句", start_ms=400, end_ms=900),
        ],
        gap_threshold_ms=0,
        minimum_duration_ms=200,
    )

    assert result.cues[0].start_ms == 100
    assert result.cues[0].end_ms == 300
    assert result.cues[0].end_ms <= result.cues[1].start_ms
    assert original.end_ms == 150
    assert result.unresolved_short_cues == []


def test_postprocess_reports_short_cue_when_no_free_time_exists() -> None:
    result = postprocess_subtitle_cues(
        [
            SubtitleCue(source_index=0, text="前句", start_ms=0, end_ms=500),
            SubtitleCue(source_index=1, text="短句", start_ms=500, end_ms=550),
            SubtitleCue(source_index=2, text="后句", start_ms=550, end_ms=1000),
        ],
        gap_threshold_ms=67,
        minimum_duration_ms=200,
    )

    assert result.unresolved_short_cues == [
        {"source_index": 1, "start_ms": 500, "end_ms": 550, "duration_ms": 50}
    ]


def test_postprocess_rejects_negative_thresholds() -> None:
    try:
        postprocess_subtitle_cues([], gap_threshold_ms=-1, minimum_duration_ms=200)
    except ValueError as error:
        assert "间隙阈值" in str(error)
    else:
        raise AssertionError("expected negative threshold to fail")


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
