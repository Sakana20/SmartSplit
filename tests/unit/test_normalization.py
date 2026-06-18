from funasr_timeline.normalization import normalize_text


def test_normalize_text_removes_punctuation_space_and_normalizes_width() -> None:
    normalized = normalize_text(" Ｅｎｇｌｉｓｈ，１２３！")

    assert normalized.text == "english123"
    assert normalized.chars[0].original_char == "Ｅ"
    assert normalized.chars[0].normalized_char == "e"
    assert 0 not in normalized.original_to_normalized
