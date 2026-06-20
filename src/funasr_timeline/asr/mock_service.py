from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from funasr_timeline.asr.base import AsrInfo, AsrService, AsrToken, AudioInfo, WordTimeline


class MockAsrService(AsrService):
    provider = "mock"

    def __init__(self, timeline_path: Path) -> None:
        self.timeline_path = timeline_path

    def transcribe(self, audio_path: Path) -> WordTimeline:
        logger.debug("读取 mock ASR 时间轴：timeline={} audio={}", self.timeline_path, audio_path)
        with self.timeline_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        audio = _load_audio_info(payload.get("audio", {}), audio_path)
        asr = _load_asr_info(payload.get("asr", {}), payload)
        tokens = [_load_token(item, index) for index, item in enumerate(payload.get("tokens", []))]
        logger.debug("mock ASR 时间轴读取完成：tokens={} provider={}", len(tokens), asr.provider)
        return WordTimeline(audio=audio, asr=asr, tokens=tokens)


def _load_audio_info(payload: Any, audio_path: Path) -> AudioInfo:
    if not isinstance(payload, dict):
        payload = {}
    raw_path = payload.get("path") or str(audio_path)
    audio_format = payload.get("format") or audio_path.suffix.lstrip(".").lower()
    duration = payload.get("duration_ms")
    return AudioInfo(
        path=str(raw_path),
        format=str(audio_format),
        duration_ms=int(duration) if duration is not None else None,
    )


def _load_asr_info(payload: Any, root: dict[str, Any]) -> AsrInfo:
    if not isinstance(payload, dict):
        payload = {}
    text = payload.get("text") or root.get("asr_text") or ""
    return AsrInfo(
        provider=str(payload.get("provider") or "mock"),
        model=str(payload["model"]) if payload.get("model") is not None else None,
        text=str(text),
    )


def _load_token(payload: Any, fallback_index: int) -> AsrToken:
    if not isinstance(payload, dict):
        raise ValueError(f"ASR token at index {fallback_index} must be an object")

    return AsrToken(
        index=int(payload.get("index", fallback_index)),
        text=str(payload["text"]),
        start_ms=int(payload["start_ms"]),
        end_ms=int(payload["end_ms"]),
        confidence=(
            float(payload["confidence"]) if payload.get("confidence") is not None else None
        ),
        source=str(payload.get("source", "asr")),
    )
