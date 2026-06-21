import json
from pathlib import Path

from funasr_timeline.cli import main


def test_cli_generates_expected_files(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    exit_code = main(
        [
            "--manuscript",
            str(fixture_dir / "manuscript.txt"),
            "--audio",
            str(fixture_dir / "audio.mp3"),
            "--output-dir",
            str(tmp_path),
            "--segmenter",
            "regex",
            "--asr-provider",
            "mock",
            "--mock-word-timeline",
            str(fixture_dir / "word_timeline.json"),
            "--timeline-provider",
            "asr-fuzzy",
            "--no-align-last-subtitle-to-audio-end",
        ]
    )

    assert exit_code == 0
    sentence_timeline_path = tmp_path / "sentence_timeline.json"
    assert sentence_timeline_path.exists()
    assert (tmp_path / "sentence_timeline.srt").exists()

    sentence_timeline = json.loads(sentence_timeline_path.read_text(encoding="utf-8"))
    assert len(sentence_timeline) == 2
    assert sentence_timeline[0]["status"] == "ok"
    assert sentence_timeline[0]["start_ms"] == 100
    assert "00:00:00,000 -->" in (tmp_path / "sentence_timeline.srt").read_text(encoding="utf-8")


def test_cli_can_run_hybrid_with_mock_forced_aligner(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    exit_code = main(
        [
            "--manuscript",
            str(fixture_dir / "manuscript.txt"),
            "--audio",
            str(fixture_dir / "audio.mp3"),
            "--output-dir",
            str(tmp_path),
            "--segmenter",
            "regex",
            "--aligner-config",
            str(fixture_dir / "forced_alignment" / "config.toml"),
            "--mock-word-timeline",
            str(fixture_dir / "word_timeline.json"),
            "--no-align-last-subtitle-to-audio-end",
            "--no-align-first-subtitle-to-audio-start",
        ]
    )

    assert exit_code == 0
    sentence_timeline = json.loads(
        (tmp_path / "sentence_timeline.json").read_text(encoding="utf-8")
    )
    assert sentence_timeline[0]["start_ms"] == 120
    assert "00:00:00,120 -->" in (tmp_path / "sentence_timeline.srt").read_text(encoding="utf-8")
    assert (tmp_path / "forced_alignment.json").exists()
    assert (tmp_path / "telemetry.json").exists()


def test_cli_can_run_segmentation_only(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    exit_code = main(
        [
            "--manuscript",
            str(fixture_dir / "manuscript.txt"),
            "--output-dir",
            str(tmp_path),
            "--segment-only",
            "--segmenter",
            "jieba-subtitle",
        ]
    )

    assert exit_code == 0
    editable_segments = (tmp_path / "editable_segments.txt").read_text(encoding="utf-8")
    assert "第一句话" in editable_segments
    assert (tmp_path / "manuscript_segments.json").exists()
