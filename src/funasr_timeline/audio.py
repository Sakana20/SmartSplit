from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
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


@dataclass(frozen=True, slots=True)
class MediaStreamTiming:
    path: Path
    stream_type: str
    stream_index: int
    start_seconds: Decimal
    end_seconds: Decimal
    duration_seconds: Decimal
    source: str
    format_start_seconds: Decimal
    format_duration_seconds: Decimal | None

    @property
    def end_milliseconds(self) -> int:
        return int((self.end_seconds * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    def to_report(self) -> dict[str, Any]:
        return {
            "media_path": str(self.path),
            "target_stream_type": self.stream_type,
            "target_stream_index": self.stream_index,
            "timing_source": self.source,
            "media_format_duration_ms": _optional_milliseconds(self.format_duration_seconds),
            "target_stream_start_ms": _decimal_milliseconds(self.start_seconds),
            "target_stream_end_ms": self.end_milliseconds,
            "target_stream_duration_ms": _decimal_milliseconds(self.duration_seconds),
            "quantization": "nearest_millisecond",
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
    """Read the first audio stream's presentation end for compatibility callers."""
    return float(probe_audio_stream_timing(audio_path).end_seconds)


def probe_audio_stream_timing(audio_path: Path) -> MediaStreamTiming:
    """Read the first audio stream's presentation timing without using video duration."""
    return _probe_media_stream_timing(audio_path, preferred_stream_type="audio")


def probe_subtitle_alignment_timing(media_path: Path) -> MediaStreamTiming:
    """Read the first video stream end, or the first audio stream for audio-only media."""
    payload = _probe_media_payload(media_path)
    streams = _payload_streams(payload)
    preferred_stream_type = (
        "video" if any(_is_target_stream(stream, "video") for stream in streams) else "audio"
    )
    return _probe_media_stream_timing(
        media_path,
        preferred_stream_type=preferred_stream_type,
        payload=payload,
    )


def _probe_media_payload(media_path: Path) -> dict[str, Any]:
    if not media_path.is_file():
        raise FileNotFoundError(f"媒体文件不存在：{media_path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=start_time,duration:"
        "stream=index,codec_type,start_time,duration,duration_ts,time_base:"
        "stream_disposition=attached_pic",
        "-of",
        "json",
        str(media_path),
    ]
    return _run_ffprobe_json(command, media_path)


def _probe_media_stream_timing(
    media_path: Path,
    *,
    preferred_stream_type: str,
    payload: dict[str, Any] | None = None,
) -> MediaStreamTiming:
    if not media_path.is_file():
        raise FileNotFoundError(f"媒体文件不存在：{media_path}")

    payload = payload or _probe_media_payload(media_path)
    streams = _payload_streams(payload)
    selected_streams = [
        stream for stream in streams if _is_target_stream(stream, preferred_stream_type)
    ]
    if not selected_streams:
        raise RuntimeError(f"媒体中没有可用{preferred_stream_type}流：{media_path}")

    stream = selected_streams[0]
    stream_index = _parse_int(stream.get("index"), field="stream.index", path=media_path)
    raw_format = payload.get("format")
    format_info = raw_format if isinstance(raw_format, dict) else {}
    parsed_format_start = _optional_decimal(format_info.get("start_time"))
    format_start = parsed_format_start if parsed_format_start is not None else Decimal(0)
    format_duration = _optional_nonnegative_decimal(format_info.get("duration"))
    parsed_stream_start = _optional_decimal(stream.get("start_time"))
    stream_start = parsed_stream_start if parsed_stream_start is not None else format_start

    duration, source = _stream_duration(stream)
    if duration is not None:
        normalized_start = stream_start - format_start
        return MediaStreamTiming(
            path=media_path,
            stream_type=preferred_stream_type,
            stream_index=stream_index,
            start_seconds=normalized_start,
            end_seconds=normalized_start + duration,
            duration_seconds=duration,
            source=source,
            format_start_seconds=format_start,
            format_duration_seconds=format_duration,
        )

    packet_end = _probe_last_stream_packet_end(
        media_path,
        format_start,
        stream_selector="v:0" if preferred_stream_type == "video" else "a:0",
    )
    if packet_end is not None:
        normalized_start = stream_start - format_start
        return MediaStreamTiming(
            path=media_path,
            stream_type=preferred_stream_type,
            stream_index=stream_index,
            start_seconds=normalized_start,
            end_seconds=packet_end,
            duration_seconds=max(Decimal(0), packet_end - normalized_start),
            source="packet_end",
            format_start_seconds=format_start,
            format_duration_seconds=format_duration,
        )

    if len(streams) == 1 and format_duration is not None:
        return MediaStreamTiming(
            path=media_path,
            stream_type=preferred_stream_type,
            stream_index=stream_index,
            start_seconds=Decimal(0),
            end_seconds=format_duration,
            duration_seconds=format_duration,
            source="format_duration_fallback",
            format_start_seconds=format_start,
            format_duration_seconds=format_duration,
        )
    raise RuntimeError(
        f"无法确定目标{preferred_stream_type}流的结束时间：{media_path} stream_index={stream_index}"
    )


def _payload_streams(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        return []
    return [stream for stream in raw_streams if isinstance(stream, dict)]


def _is_target_stream(stream: dict[str, Any], stream_type: str) -> bool:
    if stream.get("codec_type") != stream_type:
        return False
    raw_disposition = stream.get("disposition")
    disposition = raw_disposition if isinstance(raw_disposition, dict) else {}
    return stream_type != "video" or disposition.get("attached_pic") != 1


def _run_ffprobe_json(command: list[str], audio_path: Path) -> dict[str, Any]:
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

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe 返回了无效 JSON：{audio_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe 返回的顶层数据不是对象：{audio_path}")
    return payload


def _stream_duration(stream: dict[str, Any]) -> tuple[Decimal | None, str]:
    duration_ts = _optional_nonnegative_decimal(stream.get("duration_ts"))
    time_base = _optional_time_base(stream.get("time_base"))
    if duration_ts is not None and time_base is not None:
        return duration_ts * time_base, "stream_duration_ts"
    duration = _optional_nonnegative_decimal(stream.get("duration"))
    if duration is not None:
        return duration, "stream_duration"
    return None, ""


def _probe_last_stream_packet_end(
    media_path: Path,
    format_start: Decimal,
    *,
    stream_selector: str,
) -> Decimal | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        stream_selector,
        "-show_packets",
        "-show_entries",
        "packet=pts_time,dts_time,duration_time",
        "-of",
        "json",
        str(media_path),
    ]
    payload = _run_ffprobe_json(command, media_path)
    raw_packets = payload.get("packets")
    packets = raw_packets if isinstance(raw_packets, list) else []
    ends: list[Decimal] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        timestamp = _optional_decimal(packet.get("pts_time"))
        if timestamp is None:
            timestamp = _optional_decimal(packet.get("dts_time"))
        duration = _optional_nonnegative_decimal(packet.get("duration_time"))
        if timestamp is not None and duration is not None:
            ends.append(timestamp + duration - format_start)
    return max(ends) if ends else None


def _optional_time_base(value: Any) -> Decimal | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    numerator_value = _optional_decimal(numerator)
    denominator_value = _optional_decimal(denominator)
    if numerator_value is None or denominator_value is None or denominator_value == 0:
        return None
    return numerator_value / denominator_value


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _optional_nonnegative_decimal(value: Any) -> Decimal | None:
    result = _optional_decimal(value)
    return result if result is not None and result >= 0 else None


def _parse_int(value: Any, *, field: str, path: Path) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"ffprobe 返回了无效字段：{path} {field}={value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"ffprobe 返回了无效字段：{path} {field}={value!r}") from error


def _decimal_milliseconds(seconds: Decimal) -> int:
    return int((seconds * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _optional_milliseconds(seconds: Decimal | None) -> int | None:
    return _decimal_milliseconds(seconds) if seconds is not None else None


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
