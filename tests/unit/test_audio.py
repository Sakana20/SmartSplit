import json
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

import funasr_timeline.audio as audio_module
from funasr_timeline.audio import (
    prepare_audio_for_asr,
    probe_audio_duration_seconds,
    probe_audio_stream_timing,
    probe_subtitle_alignment_timing,
)


def _audio_probe_json(
    *,
    duration: str = "1.25",
    duration_ts: int = 1250,
    time_base: str = "1/1000",
) -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "time_base": time_base,
                    "start_time": "0.000000",
                    "duration_ts": duration_ts,
                    "duration": duration,
                }
            ],
            "format": {"start_time": "0.000000", "duration": duration},
        }
    )


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
        return subprocess.CompletedProcess(command, 0, stdout=_audio_probe_json(), stderr="")

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
        return subprocess.CompletedProcess(command, 0, stdout=_audio_probe_json(), stderr="")

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
        return subprocess.CompletedProcess(command, 0, stdout=_audio_probe_json(), stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    first = prepare_audio_for_asr(first_source, tmp_path / "output")
    second = prepare_audio_for_asr(second_source, tmp_path / "output")

    assert first.asr_path != second.asr_path


def test_probe_audio_duration_uses_audio_stream_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "voice.m4a"
    audio_path.write_bytes(b"m4a")

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "ffprobe"
        assert command[-1] == str(audio_path)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_audio_probe_json(duration="4.501", duration_ts=4501),
            stderr="",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    assert probe_audio_duration_seconds(audio_path) == 4.501


def test_probe_audio_stream_ignores_longer_video_and_uses_rational_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.mp4"
    media_path.write_bytes(b"mp4")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "time_base": "1/12800",
                "start_time": "0.000000",
                "duration_ts": 239616,
                "duration": "18.720000",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "time_base": "1/44100",
                "start_time": "0.000000",
                "duration_ts": 822245,
                "duration": "18.645011",
            },
        ],
        "format": {"start_time": "0.000000", "duration": "18.720000"},
    }

    monkeypatch.setattr(
        audio_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    timing = probe_audio_stream_timing(media_path)

    assert timing.stream_index == 1
    assert timing.source == "stream_duration_ts"
    assert timing.duration_seconds == Decimal(822245) / Decimal(44100)
    assert timing.end_milliseconds == 18645
    assert timing.format_duration_seconds == Decimal("18.720000")


def test_probe_subtitle_alignment_prefers_video_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.mp4"
    media_path.write_bytes(b"mp4")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "time_base": "1/12800",
                "start_time": "0.000000",
                "duration_ts": 239616,
                "duration": "18.720000",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "time_base": "1/44100",
                "start_time": "0.000000",
                "duration_ts": 822245,
                "duration": "18.645011",
            },
        ],
        "format": {"start_time": "0.000000", "duration": "18.720000"},
    }

    monkeypatch.setattr(
        audio_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    timing = probe_subtitle_alignment_timing(media_path)

    assert timing.stream_type == "video"
    assert timing.stream_index == 0
    assert timing.end_milliseconds == 18720


def test_probe_subtitle_alignment_keeps_video_when_audio_is_longer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.mp4"
    media_path.write_bytes(b"mp4")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "time_base": "1/1000",
                "duration_ts": 5000,
            },
            {
                "index": 1,
                "codec_type": "audio",
                "time_base": "1/1000",
                "duration_ts": 5500,
            },
        ],
        "format": {"duration": "5.500"},
    }

    monkeypatch.setattr(
        audio_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    timing = probe_subtitle_alignment_timing(media_path)

    assert timing.stream_type == "video"
    assert timing.end_milliseconds == 5000
    assert timing.format_duration_seconds == Decimal("5.500")


def test_probe_subtitle_alignment_ignores_attached_cover_art(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.mp3"
    media_path.write_bytes(b"mp3")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "time_base": "1/1000",
                "duration_ts": 4501,
            },
            {
                "index": 1,
                "codec_type": "video",
                "duration": "4.501",
                "disposition": {"attached_pic": 1},
            },
        ],
        "format": {"duration": "4.501"},
    }

    monkeypatch.setattr(
        audio_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    timing = probe_subtitle_alignment_timing(media_path)

    assert timing.stream_type == "audio"
    assert timing.stream_index == 0
    assert timing.end_milliseconds == 4501


def test_probe_audio_stream_falls_back_to_last_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.m4a"
    media_path.write_bytes(b"m4a")
    stream_payload = {
        "streams": [{"index": 0, "codec_type": "audio", "start_time": "0.100000"}],
        "format": {"start_time": "0.100000", "duration": "4.600000"},
    }
    packet_payload = {
        "packets": [
            {"pts_time": "4.500000", "duration_time": "0.101000"},
        ]
    }

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        payload = packet_payload if "-show_packets" in command else stream_payload
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    timing = probe_audio_stream_timing(media_path)

    assert timing.source == "packet_end"
    assert timing.start_seconds == Decimal(0)
    assert timing.end_seconds == Decimal("4.501000")
    assert timing.end_milliseconds == 4501


def test_probe_audio_stream_preserves_zero_stream_start_against_nonzero_format_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.m4a"
    media_path.write_bytes(b"m4a")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "time_base": "1/1000",
                "start_time": "0.000",
                "duration_ts": 1000,
            }
        ],
        "format": {"start_time": "0.100", "duration": "1.000"},
    }

    monkeypatch.setattr(
        audio_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    timing = probe_audio_stream_timing(media_path)

    assert timing.start_seconds == Decimal("-0.100")
    assert timing.end_seconds == Decimal("0.900")


def test_probe_audio_stream_does_not_use_container_fallback_for_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.mp4"
    media_path.write_bytes(b"mp4")
    stream_payload = {
        "streams": [
            {"index": 0, "codec_type": "video", "duration": "5.0"},
            {"index": 1, "codec_type": "audio"},
        ],
        "format": {"duration": "5.0"},
    }
    packet_payload = {"packets": []}

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        payload = packet_payload if "-show_packets" in command else stream_payload
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="无法确定目标audio流"):
        probe_audio_stream_timing(media_path)


def test_probe_audio_stream_allows_format_fallback_for_audio_only_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "voice.mp3"
    media_path.write_bytes(b"mp3")
    stream_payload = {
        "streams": [{"index": 0, "codec_type": "audio"}],
        "format": {"start_time": "0.025", "duration": "4.501"},
    }
    packet_payload = {"packets": []}

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        payload = packet_payload if "-show_packets" in command else stream_payload
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    timing = probe_audio_stream_timing(media_path)

    assert timing.source == "format_duration_fallback"
    assert timing.end_milliseconds == 4501


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
