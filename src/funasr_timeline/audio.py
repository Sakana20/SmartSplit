from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class AudioPreparation:
    source_path: Path
    asr_path: Path
    converted: bool
    conversion_reused: bool = False
    conversion_cache_key: str | None = None
    ffmpeg_command: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_audio_path": str(self.source_path),
            "asr_audio_path": str(self.asr_path),
            "source_format": self.source_path.suffix.lstrip(".").lower(),
            "asr_format": self.asr_path.suffix.lstrip(".").lower(),
            "converted": self.converted,
            "conversion_reused": self.conversion_reused,
            "conversion_cache_key": self.conversion_cache_key,
            "ffmpeg_command": self.ffmpeg_command,
        }


def prepare_audio_for_asr(audio_path: Path, output_dir: Path) -> AudioPreparation:
    """Ensure the audio sent to ASR is an MP3, converting with ffmpeg if needed."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    if audio_path.suffix.lower() == ".mp3":
        return AudioPreparation(
            source_path=audio_path,
            asr_path=audio_path,
            converted=False,
        )

    conversion_dir = output_dir / "audio"
    conversion_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _conversion_cache_key(audio_path)
    converted_path = conversion_dir / f"{audio_path.stem}-{cache_key[:12]}.mp3"
    if converted_path.is_file() and converted_path.stat().st_size > 0:
        try:
            probe_audio_duration_seconds(converted_path)
        except RuntimeError:
            pass
        else:
            return AudioPreparation(
                source_path=audio_path,
                asr_path=converted_path,
                converted=True,
                conversion_reused=True,
                conversion_cache_key=cache_key,
            )

    temporary_path = conversion_dir / (f".{converted_path.stem}.{uuid4().hex}.temporary.mp3")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(temporary_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "非 MP3 音频需要调用 ffmpeg 转换，但当前环境找不到 ffmpeg；"
            "请先安装 ffmpeg 并确保它在 PATH 中。"
        ) from error
    except subprocess.CalledProcessError as error:
        temporary_path.unlink(missing_ok=True)
        stderr = (error.stderr or "").strip()
        detail = stderr[-2000:] if stderr else "无 ffmpeg 错误输出"
        raise RuntimeError(f"ffmpeg 音频转换失败：{audio_path}\n{detail}") from error

    try:
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg 音频转换未生成有效文件：{temporary_path}")
        try:
            probe_audio_duration_seconds(temporary_path)
        except RuntimeError as error:
            raise RuntimeError(f"ffmpeg 生成的 MP3 无法读取：{temporary_path}") from error
        temporary_path.replace(converted_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return AudioPreparation(
        source_path=audio_path,
        asr_path=converted_path,
        converted=True,
        conversion_cache_key=cache_key,
        ffmpeg_command=command,
    )


def probe_audio_duration_seconds(audio_path: Path) -> float:
    """Read an audio duration with ffprobe for every ffmpeg-supported input format."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "读取音频时长需要调用 ffprobe，但当前环境找不到 ffprobe；"
            "请安装完整的 ffmpeg 工具并确保 ffprobe 位于 PATH 中。"
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        detail = stderr[-2000:] if stderr else "无 ffprobe 错误输出"
        raise RuntimeError(f"ffprobe 无法读取音频：{audio_path}\n{detail}") from error

    raw_duration = completed.stdout.strip()
    try:
        duration_seconds = float(raw_duration)
    except ValueError as error:
        raise RuntimeError(
            f"ffprobe 返回了无效的音频时长：{audio_path} duration={raw_duration!r}"
        ) from error
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise RuntimeError(f"ffprobe 返回了无效的音频时长：{audio_path} duration={raw_duration!r}")
    return duration_seconds


def _conversion_cache_key(audio_path: Path) -> str:
    stat = audio_path.stat()
    fingerprint = "\0".join(
        [
            str(audio_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ]
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
