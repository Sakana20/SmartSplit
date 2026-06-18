from pathlib import Path

import pytest

from funasr_timeline.asr.paraformer_zh_service import paraformer_zh_result_to_timeline


def test_paraformer_zh_result_to_timeline_maps_spaced_text_to_timestamps() -> None:
    timeline = paraformer_zh_result_to_timeline(
        result={
            "text": "正 是 因 为",
            "timestamp": [[410, 650], [650, 830], [830, 990], [990, 1150]],
        },
        audio_path=Path("input.mp3"),
        model_dir=Path("/models/paraformer"),
    )

    assert timeline.audio.format == "mp3"
    assert timeline.audio.duration_ms == 1150
    assert timeline.asr.provider == "paraformer-zh"
    assert timeline.asr.model == "paraformer-zh:/models/paraformer"
    assert [token.text for token in timeline.tokens] == ["正", "是", "因", "为"]
    assert timeline.tokens[0].start_ms == 410
    assert timeline.tokens[-1].end_ms == 1150


def test_paraformer_zh_result_to_timeline_ignores_punctuation_when_timestamps_omit_it() -> None:
    timeline = paraformer_zh_result_to_timeline(
        result={
            "text": "你好，世界。",
            "timestamp": [[0, 100], [100, 200], [200, 300], [300, 400]],
        },
        audio_path=Path("input.mp3"),
        model_dir=Path("/models/paraformer"),
    )

    assert [token.text for token in timeline.tokens] == ["你", "好", "世", "界"]


def test_paraformer_zh_result_to_timeline_requires_matching_timestamp_count() -> None:
    with pytest.raises(ValueError, match="数量无法对应"):
        paraformer_zh_result_to_timeline(
            result={
                "text": "你好",
                "timestamp": [[0, 100]],
            },
            audio_path=Path("input.mp3"),
            model_dir=Path("/models/paraformer"),
        )
