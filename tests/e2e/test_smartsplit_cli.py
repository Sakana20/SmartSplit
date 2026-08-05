import json
from pathlib import Path

import pytest

from funasr_timeline.cli import build_parser
from funasr_timeline.smartsplit_cli import (
    build_smartsplit_defaults,
    load_env_file,
    main,
)


def test_smartsplit_defaults_match_skill_profile() -> None:
    defaults = build_smartsplit_defaults({})
    args = build_parser(defaults).parse_args(
        ["--manuscript", "input.txt", "--output-dir", "output"]
    )

    assert args.segmenter == "llm"
    assert args.llm_fallback_segmenter == "hanlp"
    assert args.timeline_provider == "hybrid"
    assert args.asr_provider == "paraformer-zh"
    assert args.paraformer_model_dir == Path("/Users/sakana/PyEnv/paraformer")
    assert args.paraformer_device == "mps"
    assert args.log_level == "INFO"
    assert args.subtitle_gap_threshold_ms == 667
    assert args.subtitle_min_duration_ms == 200


def test_smartsplit_environment_precedence_and_cli_override() -> None:
    defaults = build_smartsplit_defaults(
        {
            "FUNASR_TIMELINE_E2E_LLM_CONFIG": "from-e2e.toml",
            "SMARTSPLIT_LLM_CONFIG": "from-smartsplit.toml",
            "FUNASR_TIMELINE_E2E_ALIGNER_CONFIG": "aligner.toml",
            "SMARTSPLIT_PARAFORMER_DEVICE": "cpu",
        }
    )
    args = build_parser(defaults).parse_args(
        [
            "--manuscript",
            "input.txt",
            "--output-dir",
            "output",
            "--segmenter",
            "regex",
            "--llm-config",
            "from-cli.toml",
            "--paraformer-device",
            "cuda:0",
        ]
    )

    assert defaults.llm_config == Path("from-smartsplit.toml")
    assert defaults.aligner_config == Path("aligner.toml")
    assert defaults.paraformer_device == "cpu"
    assert args.segmenter == "regex"
    assert args.llm_config == Path("from-cli.toml")
    assert args.paraformer_device == "cuda:0"


def test_load_env_file_is_non_executing_and_preserves_process_environment(tmp_path: Path) -> None:
    env_path = tmp_path / "smartsplit.env"
    env_path.write_text(
        "# comment\nEXISTING=from-file\nPLAIN=value\nQUOTED='two words'\n",
        encoding="utf-8",
    )
    environ = {"EXISTING": "from-process"}

    load_env_file(env_path, environ)

    assert environ == {
        "EXISTING": "from-process",
        "PLAIN": "value",
        "QUOTED": "two words",
    }


def test_smartsplit_can_run_segmentation_with_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "smartsplit.env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SMARTSPLIT_E2E_ENV", str(env_path))
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "--manuscript",
            "tests/fixtures/manuscript.txt",
            "--output-dir",
            str(output_dir),
            "--segment-only",
            "--segmenter",
            "regex",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "editable_segments.txt").exists()
    assert (output_dir / "manuscript_segments.json").exists()


def test_smartsplit_can_override_profile_for_mock_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "smartsplit.env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("SMARTSPLIT_E2E_ENV", str(env_path))
    fixture_dir = Path("tests/fixtures")
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "--manuscript",
            str(fixture_dir / "manuscript.txt"),
            "--audio",
            str(fixture_dir / "audio.mp3"),
            "--output-dir",
            str(output_dir),
            "--segmenter",
            "regex",
            "--timeline-provider",
            "asr-fuzzy",
            "--asr-provider",
            "mock",
            "--mock-word-timeline",
            str(fixture_dir / "word_timeline.json"),
            "--aligner-config",
            str(fixture_dir / "forced_alignment" / "config.toml"),
            "--no-align-last-subtitle-to-audio-end",
        ]
    )

    assert exit_code == 0
    timeline = json.loads((output_dir / "sentence_timeline.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in timeline] == ["第一句话。", "第二段有Ｅｎｇｌｉｓｈ 123！"]
    assert (output_dir / "sentence_timeline.srt").exists()


def test_smartsplit_reports_missing_explicit_environment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_path = tmp_path / "missing.env"
    monkeypatch.setenv("SMARTSPLIT_E2E_ENV", str(missing_path))

    assert main([]) == 2
    assert str(missing_path) in capsys.readouterr().err
