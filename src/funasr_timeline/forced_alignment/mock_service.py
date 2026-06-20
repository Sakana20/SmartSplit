from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from funasr_timeline.asr.base import AudioInfo
from funasr_timeline.forced_alignment.base import (
    ForcedAlignmentInfo,
    ForcedAlignmentResult,
    ForcedAlignmentService,
    ForcedAlignmentUnit,
)
from funasr_timeline.normalization import normalize_text


class MockForcedAlignmentService(ForcedAlignmentService):
    provider = "mock-forced"

    def __init__(self, units_path: Path) -> None:
        self.units_path = units_path

    def align(self, audio_path: Path, text: str, language: str) -> ForcedAlignmentResult:
        logger.debug("读取 mock forced alignment：units={} audio={}", self.units_path, audio_path)
        payload = json.loads(self.units_path.read_text(encoding="utf-8"))
        audio_payload = payload.get("audio") if isinstance(payload, dict) else None
        aligner_payload = payload.get("aligner") if isinstance(payload, dict) else None
        raw_units = payload.get("units", []) if isinstance(payload, dict) else []

        units = [_load_unit(item, index) for index, item in enumerate(raw_units)]
        normalized_text = normalize_text(text).text
        forced_normalized_text = "".join(unit.normalized_text for unit in units)
        return ForcedAlignmentResult(
            audio=_load_audio_info(audio_payload, audio_path, units),
            aligner=_load_aligner_info(aligner_payload, language),
            input_text=text,
            normalized_text=normalized_text,
            forced_normalized_text=forced_normalized_text,
            normalized_text_match=normalized_text == forced_normalized_text,
            units=units,
            diagnostics={
                "source": str(self.units_path),
                "unit_count": len(units),
            },
        )


def _load_audio_info(
    payload: Any,
    audio_path: Path,
    units: list[ForcedAlignmentUnit],
) -> AudioInfo:
    if not isinstance(payload, dict):
        payload = {}
    duration_ms = payload.get("duration_ms")
    if duration_ms is None and units:
        duration_ms = max(unit.end_ms for unit in units)
    return AudioInfo(
        path=str(payload.get("path") or audio_path),
        format=str(payload.get("format") or audio_path.suffix.lstrip(".").lower()),
        duration_ms=int(duration_ms) if duration_ms is not None else None,
    )


def _load_aligner_info(payload: Any, language: str) -> ForcedAlignmentInfo:
    if not isinstance(payload, dict):
        payload = {}
    return ForcedAlignmentInfo(
        provider=str(payload.get("provider") or "mock-forced"),
        model=str(payload["model"]) if payload.get("model") is not None else None,
        device_map=str(payload["device_map"]) if payload.get("device_map") is not None else None,
        dtype=str(payload["dtype"]) if payload.get("dtype") is not None else None,
        language=str(payload.get("language") or language),
    )


def _load_unit(payload: Any, fallback_index: int) -> ForcedAlignmentUnit:
    if not isinstance(payload, dict):
        raise ValueError(f"forced alignment unit at index {fallback_index} must be an object")
    text = str(payload["text"])
    normalized_text = str(payload.get("normalized_text") or normalize_text(text).text)
    return ForcedAlignmentUnit(
        index=int(payload.get("index", fallback_index)),
        text=text,
        start_ms=int(payload["start_ms"]),
        end_ms=int(payload["end_ms"]),
        normalized_text=normalized_text,
    )
