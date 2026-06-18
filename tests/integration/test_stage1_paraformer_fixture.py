import json
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path("tests/fixtures/stage1_paraformer")


def _read_json(name: str) -> Any:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_stage1_paraformer_fixture_contains_real_model_outputs() -> None:
    word_timeline = _read_json("word_timeline.json")
    sentence_timeline = _read_json("sentence_timeline.json")
    alignment = _read_json("alignment.json")
    report = _read_json("alignment_report.json")

    assert (FIXTURE_DIR / "audio.mp3").exists()
    assert (FIXTURE_DIR / "manuscript.txt").exists()

    assert word_timeline["asr"]["provider"] == "paraformer-zh"
    assert word_timeline["audio"]["format"] == "mp3"
    assert word_timeline["audio"]["duration_ms"] == 12925
    assert len(word_timeline["tokens"]) == 65
    assert word_timeline["tokens"][0]["text"] == "正"
    assert word_timeline["tokens"][-1]["text"] == "义"

    assert len(sentence_timeline) == 3
    assert {item["status"] for item in sentence_timeline} == {"ok"}
    assert [item["asr_token_range"] for item in sentence_timeline] == [
        [0, 22],
        [23, 48],
        [49, 64],
    ]

    assert alignment["global_match_score"] == 1.0
    assert alignment["unmatched_manuscript_indexes"] == []
    assert alignment["unmapped_asr_indexes"] == []

    assert report["asr"]["provider"] == "paraformer-zh"
    assert report["segmentation"]["sentence_count"] == 3
    assert report["low_confidence_sentences"] == []
