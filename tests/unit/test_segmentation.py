from funasr_timeline.normalization import normalize_text
from funasr_timeline.segmentation import (
    NO_SPLIT_END,
    NO_SPLIT_START,
    JiebaSubtitleSegmenter,
    RegexSentenceSegmenter,
    attach_normalized_ranges,
    available_segmenters,
    create_segmenter,
    export_editable_segments,
    load_editable_segments,
)


def test_segmenter_factory_creates_regex_segmenter() -> None:
    assert available_segmenters() == ("jieba-subtitle", "regex")
    assert isinstance(create_segmenter("regex"), RegexSentenceSegmenter)
    assert isinstance(create_segmenter("jieba-subtitle"), JiebaSubtitleSegmenter)


def test_regex_segmenter_splits_by_strong_punctuation_and_paragraphs() -> None:
    text = "第一句。第二句！\n第三段"
    result = RegexSentenceSegmenter().segment(text)
    segments = result.segments

    assert [segment.text for segment in segments] == ["第一句。", "第二句！", "第三段"]
    assert [segment.paragraph_index for segment in segments] == [0, 0, 1]
    assert [segment.boundary for segment in segments] == ["punctuation", "punctuation", "paragraph"]


def test_attach_normalized_ranges_tracks_sentence_offsets() -> None:
    text = "第一句。\n第二句。"
    normalized = normalize_text(text)
    result = RegexSentenceSegmenter().segment(text)
    segments = attach_normalized_ranges(result.segments, normalized)

    assert segments[0].normalized_text == "第一句"
    assert segments[0].normalized_start == 0
    assert segments[0].normalized_end == 3
    assert segments[1].normalized_text == "第二句"
    assert segments[1].normalized_start == 3
    assert segments[1].normalized_end == 6


def test_no_split_markers_keep_protected_content_as_one_segment() -> None:
    text = f"开头。{NO_SPLIT_START}这里。不要切！保持整体。{NO_SPLIT_END}结尾。"

    result = RegexSentenceSegmenter().segment(text)

    assert NO_SPLIT_START not in result.text
    assert NO_SPLIT_END not in result.text
    assert [segment.text for segment in result.segments] == [
        "开头。",
        "这里。不要切！保持整体。",
        "结尾。",
    ]
    assert [segment.paragraph_index for segment in result.segments] == [0, 0, 0]
    assert result.segments[1].boundary == "protected"


def test_jieba_subtitle_segmenter_keeps_words_together_under_target_length() -> None:
    text = "淘宝闪购最高红包特别划算。"

    result = JiebaSubtitleSegmenter(max_chars=5).segment(text)

    assert [segment.text for segment in result.segments] == [
        "淘宝闪购",
        "最高红包",
        "特别划算",
    ]
    assert all(len(normalize_text(segment.text).text) <= 5 for segment in result.segments)


def test_jieba_subtitle_segmenter_matches_short_video_phrase_style() -> None:
    text = (
        "说实话，荔枝真的是夏天幸福感特别高的水果，冰箱里放一盒，"
        "想吃的时候拿几个，甜甜的特别解馋。尤其最近正是季节，错过又得等一年。"
        f"{NO_SPLIT_START}现在淘宝闪购有最高12元无门槛红包{NO_SPLIT_END}"
        "，喜欢吃荔枝的直接点视频下方链接。"
    )

    result = JiebaSubtitleSegmenter().segment(text)

    assert [segment.text for segment in result.segments] == [
        "说实话",
        "荔枝真的是夏天幸福感",
        "特别高的水果",
        "冰箱里放一盒",
        "想吃的时候拿几个",
        "甜甜的",
        "特别解馋",
        "尤其最近正是季节",
        "错过又得等一年",
        "现在淘宝闪购有最高12元无门槛红包",
        "喜欢吃荔枝的",
        "直接点视频下方链接",
    ]
    assert result.segments[9].boundary == "protected"


def test_editable_segments_round_trip(tmp_path) -> None:
    result = RegexSentenceSegmenter().segment("第一句。第二句。\n第三句。")
    editable = export_editable_segments(result.segments)
    path = tmp_path / "segments.txt"
    path.write_text(editable, encoding="utf-8")

    loaded = load_editable_segments(path)

    assert editable == "第一句。\n第二句。\n\n第三句。\n"
    assert loaded.text == "第一句。第二句。\n第三句。"
    assert [segment.boundary for segment in loaded.segments] == ["editable", "editable", "editable"]
