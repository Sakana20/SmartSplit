from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from funasr_timeline.asr.base import AudioInfo


@dataclass(frozen=True, slots=True)
class ForcedAlignmentInfo:
    provider: str
    model: str | None
    device_map: str | None
    dtype: str | None
    language: str


@dataclass(frozen=True, slots=True)
class ForcedAlignmentUnit:
    index: int
    text: str
    start_ms: int
    end_ms: int
    normalized_text: str = ""


@dataclass(frozen=True, slots=True)
class ForcedAlignmentResult:
    audio: AudioInfo
    aligner: ForcedAlignmentInfo
    input_text: str
    normalized_text: str
    forced_normalized_text: str
    normalized_text_match: bool
    units: list[ForcedAlignmentUnit]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForcedAlignmentService(Protocol):
    provider: str

    def align(self, audio_path: Path, text: str, language: str) -> ForcedAlignmentResult:
        """Return forced alignment units for the given audio and ground-truth text."""
