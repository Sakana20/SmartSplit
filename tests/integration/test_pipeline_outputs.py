import json
from pathlib import Path

from funasr_timeline.asr.mock_service import MockAsrService
from funasr_timeline.pipeline import run_pipeline, run_segmentation
from funasr_timeline.segmentation import create_segmenter


def test_pipeline_writes_rich_outputs(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    paths = run_pipeline(
        manuscript_path=fixture_dir / "manuscript.txt",
        audio_path=fixture_dir / "audio.mp3",
        output_dir=tmp_path,
        asr_service=MockAsrService(fixture_dir / "word_timeline.json"),
        segmenter=create_segmenter("regex"),
    )

    assert set(paths) == {
        "word_timeline",
        "manuscript_segments",
        "normalized_text",
        "alignment",
        "sentence_timeline",
        "sentence_timeline_srt",
        "alignment_report",
    }
    for path in paths.values():
        assert path.exists()

    sentence_timeline = json.loads(paths["sentence_timeline"].read_text(encoding="utf-8"))
    assert [item["text"] for item in sentence_timeline] == [
        "第一句话。",
        "第二段有Ｅｎｇｌｉｓｈ 123！",
    ]
    assert sentence_timeline[0]["start_ms"] == 100
    assert sentence_timeline[0]["end_ms"] == 500
    assert sentence_timeline[1]["start_ms"] == 600
    assert sentence_timeline[1]["end_ms"] == 2000

    srt = paths["sentence_timeline_srt"].read_text(encoding="utf-8")
    assert "00:00:00,100 --> 00:00:00,500" in srt
    assert "第一句话。" in srt

    report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    assert report["segmentation"]["strategy"] == "regex"
    assert report["alignment"]["global_match_score"] == 1.0
    assert report["alignment"]["unmapped_asr_tokens"][0]["token_text"] == "嗯"


def test_segmentation_can_run_standalone_and_feed_pipeline(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")
    segmentation_dir = tmp_path / "segmentation"
    output_dir = tmp_path / "pipeline"

    segmentation_paths = run_segmentation(
        manuscript_path=fixture_dir / "manuscript.txt",
        output_dir=segmentation_dir,
        segmenter=create_segmenter("jieba-subtitle"),
    )

    editable_segments = segmentation_paths["editable_segments"].read_text(encoding="utf-8")
    assert "第一句话" in editable_segments

    paths = run_pipeline(
        manuscript_path=fixture_dir / "manuscript.txt",
        audio_path=fixture_dir / "audio.mp3",
        output_dir=output_dir,
        asr_service=MockAsrService(fixture_dir / "word_timeline.json"),
        segments_path=segmentation_paths["editable_segments"],
    )

    report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    assert report["segmentation"]["strategy"].startswith("editable:")
    assert paths["sentence_timeline_srt"].exists()
