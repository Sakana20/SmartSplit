import json
import subprocess
from pathlib import Path

import pytest

import funasr_timeline.audio as audio_module
from funasr_timeline.asr.base import WordTimeline
from funasr_timeline.asr.mock_service import MockAsrService
from funasr_timeline.forced_alignment.mock_service import MockForcedAlignmentService
from funasr_timeline.pipeline import run_pipeline, run_segmentation
from funasr_timeline.segmentation.factory import create_segmenter


def test_pipeline_writes_rich_outputs(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    paths = run_pipeline(
        manuscript_path=fixture_dir / "manuscript.txt",
        audio_path=fixture_dir / "audio.mp3",
        output_dir=tmp_path,
        asr_service=MockAsrService(fixture_dir / "word_timeline.json"),
        segmenter=create_segmenter("regex"),
        timeline_provider="asr-fuzzy",
    )

    assert set(paths) == {
        "word_timeline",
        "audio_conversion",
        "manuscript_segments",
        "normalized_text",
        "alignment",
        "sentence_timeline",
        "sentence_timeline_srt",
        "subtitle_render_report",
        "alignment_report",
    }
    for path in paths.values():
        assert path.exists()

    audio_conversion = json.loads(paths["audio_conversion"].read_text(encoding="utf-8"))
    assert audio_conversion["converted"] is False
    assert audio_conversion["asr_format"] == "mp3"

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
    assert "00:00:00,000 --> 00:00:00,600" in srt
    assert "第一句话。" in srt

    render_report = json.loads(paths["subtitle_render_report"].read_text(encoding="utf-8"))
    assert render_report["config"] == {
        "gap_threshold_ms": 667,
        "minimum_duration_ms": 200,
    }

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
        timeline_provider="asr-fuzzy",
    )

    report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    assert report["segmentation"]["strategy"].startswith("editable:")
    assert paths["sentence_timeline_srt"].exists()


def test_pipeline_converts_non_mp3_before_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_dir = Path("tests/fixtures")
    source_path = tmp_path / "voice.ogg"
    output_dir = tmp_path / "pipeline"
    source_path.write_bytes(b"ogg")
    asr_service = _RecordingAsrService(fixture_dir / "word_timeline.json")

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"converted mp3")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="3.0\n", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    paths = run_pipeline(
        manuscript_path=fixture_dir / "manuscript.txt",
        audio_path=source_path,
        output_dir=output_dir,
        asr_service=asr_service,
        segmenter=create_segmenter("regex"),
        timeline_provider="asr-fuzzy",
    )

    converted_path = asr_service.received_audio_path
    assert converted_path is not None
    assert converted_path.parent == output_dir / "audio"
    assert converted_path.name.startswith("voice-")
    assert converted_path.suffix == ".mp3"
    conversion = json.loads(paths["audio_conversion"].read_text(encoding="utf-8"))
    assert conversion["converted"] is True
    assert conversion["conversion_reused"] is False
    assert conversion["source_format"] == "ogg"
    assert conversion["asr_audio_path"] == str(converted_path)


def test_pipeline_writes_conversion_report_before_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_dir = Path("tests/fixtures")
    source_path = tmp_path / "voice.ogg"
    output_dir = tmp_path / "pipeline"
    source_path.write_bytes(b"ogg")

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"converted mp3")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="3.0\n", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="需要 asr_service"):
        run_pipeline(
            manuscript_path=fixture_dir / "manuscript.txt",
            audio_path=source_path,
            output_dir=output_dir,
            asr_service=None,
            segmenter=create_segmenter("regex"),
            timeline_provider="asr-fuzzy",
        )

    conversion_path = output_dir / "audio_conversion.json"
    assert conversion_path.exists()
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    assert conversion["converted"] is True


def test_pipeline_hybrid_uses_forced_timing_and_writes_telemetry(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    paths = run_pipeline(
        manuscript_path=fixture_dir / "manuscript.txt",
        audio_path=fixture_dir / "audio.mp3",
        output_dir=tmp_path,
        asr_service=MockAsrService(fixture_dir / "word_timeline.json"),
        forced_alignment_service=MockForcedAlignmentService(
            fixture_dir / "forced_alignment" / "units.json"
        ),
        segmenter=create_segmenter("regex"),
        timeline_provider="hybrid",
    )

    assert "forced_alignment" in paths
    assert "telemetry" in paths
    sentence_timeline = json.loads(paths["sentence_timeline"].read_text(encoding="utf-8"))
    assert sentence_timeline[0]["start_ms"] == 120
    assert sentence_timeline[0]["end_ms"] == 520
    assert sentence_timeline[0]["matched_token_indexes"] == [1, 2, 3, 4]
    assert sentence_timeline[0]["diagnostics"]["primary_timing_source"] == "qwen3-forced"
    assert sentence_timeline[0]["diagnostics"]["asr_fuzzy"]["start_ms"] == 100

    telemetry = json.loads(paths["telemetry"].read_text(encoding="utf-8"))
    assert telemetry["timeline_provider"] == "hybrid"
    assert telemetry["forced_alignment"]["normalized_text_match"] is True
    assert telemetry["sentences"][0]["comparison"]["start_delta_ms"] == 20

    forced_alignment = json.loads(paths["forced_alignment"].read_text(encoding="utf-8"))
    assert forced_alignment["aligner"]["provider"] == "mock-forced"
    assert forced_alignment["units"][8]["normalized_text"] == "english"


def test_pipeline_qwen3_forced_does_not_run_asr_branch(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")

    paths = run_pipeline(
        manuscript_path=fixture_dir / "manuscript.txt",
        audio_path=fixture_dir / "audio.mp3",
        output_dir=tmp_path,
        asr_service=_ExplodingAsrService(),
        forced_alignment_service=MockForcedAlignmentService(
            fixture_dir / "forced_alignment" / "units.json"
        ),
        segmenter=create_segmenter("regex"),
        timeline_provider="qwen3-forced",
    )

    sentence_timeline = json.loads(paths["sentence_timeline"].read_text(encoding="utf-8"))
    assert sentence_timeline[0]["start_ms"] == 120
    telemetry = json.loads(paths["telemetry"].read_text(encoding="utf-8"))
    assert telemetry["asr_fuzzy"]["provider"] == "none"


class _ExplodingAsrService:
    provider = "exploding"

    def transcribe(self, audio_path: Path) -> WordTimeline:
        raise AssertionError(f"ASR should not run for qwen3-forced: {audio_path}")


class _RecordingAsrService:
    provider = "recording"

    def __init__(self, timeline_path: Path) -> None:
        self.delegate = MockAsrService(timeline_path)
        self.received_audio_path: Path | None = None

    def transcribe(self, audio_path: Path) -> WordTimeline:
        self.received_audio_path = audio_path
        return self.delegate.transcribe(audio_path)
