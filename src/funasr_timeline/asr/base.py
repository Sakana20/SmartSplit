from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class AudioInfo:
    path: str
    format: str
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AsrInfo:
    provider: str
    model: str | None
    text: str


@dataclass(frozen=True, slots=True)
class AsrToken:
    index: int
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    source: str = "asr"


@dataclass(frozen=True, slots=True)
class WordTimeline:
    audio: AudioInfo
    asr: AsrInfo
    tokens: list[AsrToken]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AsrService(Protocol):
    provider: str

    def transcribe(self, audio_path: Path) -> WordTimeline:
        """Return a normalized token timeline for an audio file."""
