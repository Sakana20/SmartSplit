import subprocess
from pathlib import Path

import pytest

import funasr_timeline.audio as audio_module
from funasr_timeline.audio import prepare_audio_for_asr, probe_audio_duration_seconds


def test_prepare_audio_keeps_mp3_without_conversion(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"mp3")

    preparation = prepare_audio_for_asr(audio_path, tmp_path / "output")

    assert preparation.source_path == audio_path
    assert preparation.asr_path == audio_path
    assert preparation.converted is False
    assert preparation.conversion_reused is False
    assert preparation.ffmpeg_command is None


def test_prepare_audio_converts_non_mp3_with_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "voice.ogg"
    source_path.write_bytes(b"ogg")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True
        commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"converted mp3")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command[0] == "ffprobe"
        return subprocess.CompletedProcess(command, 0, stdout="1.25\n", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    preparation = prepare_audio_for_asr(source_path, tmp_path / "output")

    assert preparation.converted is True
    assert preparation.conversion_reused is False
    assert preparation.asr_path.parent == tmp_path / "output" / "audio"
    assert preparation.asr_path.name.startswith("voice-")
    assert preparation.asr_path.suffix == ".mp3"
    assert preparation.asr_path.read_bytes() == b"converted mp3"
    ffmpeg_commands = [command for command in commands if command[0] == "ffmpeg"]
    assert len(ffmpeg_commands) == 1
    assert "-nostdin" in ffmpeg_commands[0]
    assert "-y" not in ffmpeg_commands[0]
    assert ffmpeg_commands[0][-1] != str(preparation.asr_path)
    assert preparation.ffmpeg_command == ffmpeg_commands[0]


def test_prepare_audio_reuses_valid_cached_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "voice.ogg"
    source_path.write_bytes(b"ogg")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"converted mp3")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="1.25\n", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    first = prepare_audio_for_asr(source_path, tmp_path / "output")
    second = prepare_audio_for_asr(source_path, tmp_path / "output")

    assert second.asr_path == first.asr_path
    assert second.conversion_reused is True
    assert second.ffmpeg_command is None
    assert sum(command[0] == "ffmpeg" for command in commands) == 1


def test_prepare_audio_uses_distinct_paths_for_same_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_source = tmp_path / "first" / "voice.ogg"
    second_source = tmp_path / "second" / "voice.ogg"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"converted mp3")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="1.0\n", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    first = prepare_audio_for_asr(first_source, tmp_path / "output")
    second = prepare_audio_for_asr(second_source, tmp_path / "output")

    assert first.asr_path != second.asr_path


def test_probe_audio_duration_uses_ffprobe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "voice.m4a"
    audio_path.write_bytes(b"m4a")

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "ffprobe"
        assert command[-1] == str(audio_path)
        return subprocess.CompletedProcess(command, 0, stdout="4.501\n", stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    assert probe_audio_duration_seconds(audio_path) == 4.501


def test_prepare_audio_reports_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "voice.ogg"
    source_path.write_bytes(b"ogg")

    def missing_ffmpeg(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(audio_module.subprocess, "run", missing_ffmpeg)

    with pytest.raises(RuntimeError, match="找不到 ffmpeg"):
        prepare_audio_for_asr(source_path, tmp_path / "output")
