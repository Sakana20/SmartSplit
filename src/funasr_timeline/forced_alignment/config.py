from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from funasr_timeline.asr.paraformer_zh_service import DEFAULT_PARAFORMER_MODEL_DIR
from funasr_timeline.forced_alignment.qwen3_service import (
    DEFAULT_QWEN3_FORCED_ALIGNER_MODEL_DIR,
)

TimelineProvider = Literal["asr-fuzzy", "qwen3-forced", "hybrid"]


@dataclass(frozen=True, slots=True)
class TimelineConfig:
    provider: TimelineProvider = "hybrid"
    primary: str = "qwen3-forced"


@dataclass(frozen=True, slots=True)
class Qwen3ForcedConfig:
    provider: str = "qwen3-forced"
    model_dir: Path = DEFAULT_QWEN3_FORCED_ALIGNER_MODEL_DIR
    device_map: str = "mps"
    dtype: str = "bfloat16"
    language: str = "Chinese"
    max_audio_seconds: int = 300
    units_path: Path | None = None


@dataclass(frozen=True, slots=True)
class AsrConfig:
    provider: str = "paraformer-zh"


@dataclass(frozen=True, slots=True)
class ParaformerZhConfig:
    model_dir: Path = DEFAULT_PARAFORMER_MODEL_DIR
    device: str = "mps"


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    include_forced_units: bool = True
    include_asr_tokens: bool = True
    include_sentence_comparison: bool = True


@dataclass(frozen=True, slots=True)
class AlignerConfig:
    timeline: TimelineConfig
    qwen3_forced: Qwen3ForcedConfig
    asr: AsrConfig
    paraformer_zh: ParaformerZhConfig
    telemetry: TelemetryConfig


def load_aligner_config(path: Path | None) -> AlignerConfig:
    payload: dict[str, Any] = {}
    if path is not None and path.exists():
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    elif path is not None and path.name.endswith(".local.toml"):
        payload = {}
    elif path is not None:
        raise FileNotFoundError(f"aligner 配置文件不存在：{path}")

    return AlignerConfig(
        timeline=_load_timeline(payload.get("timeline", {})),
        qwen3_forced=_load_qwen3_forced(payload.get("qwen3_forced", {})),
        asr=_load_asr(payload.get("asr", {})),
        paraformer_zh=_load_paraformer_zh(payload.get("paraformer_zh", {})),
        telemetry=_load_telemetry(payload.get("telemetry", {})),
    )


def _load_timeline(payload: Any) -> TimelineConfig:
    if not isinstance(payload, dict):
        payload = {}
    provider = str(payload.get("provider") or "hybrid")
    if provider not in {"asr-fuzzy", "qwen3-forced", "hybrid"}:
        raise ValueError(f"不支持的 timeline.provider：{provider}")
    return TimelineConfig(
        provider=provider,  # type: ignore[arg-type]
        primary=str(payload.get("primary") or "qwen3-forced"),
    )


def _load_qwen3_forced(payload: Any) -> Qwen3ForcedConfig:
    if not isinstance(payload, dict):
        payload = {}
    units_path = payload.get("units_path")
    return Qwen3ForcedConfig(
        provider=str(payload.get("provider") or "qwen3-forced"),
        model_dir=Path(str(payload.get("model_dir") or DEFAULT_QWEN3_FORCED_ALIGNER_MODEL_DIR)),
        device_map=str(payload.get("device_map") or "mps"),
        dtype=str(payload.get("dtype") or "bfloat16"),
        language=str(payload.get("language") or "Chinese"),
        max_audio_seconds=int(payload.get("max_audio_seconds") or 300),
        units_path=Path(str(units_path)) if units_path is not None else None,
    )


def _load_asr(payload: Any) -> AsrConfig:
    if not isinstance(payload, dict):
        payload = {}
    provider = str(payload.get("provider") or "paraformer-zh")
    if provider not in {"mock", "paraformer-zh"}:
        raise ValueError(f"不支持的 asr.provider：{provider}")
    return AsrConfig(provider=provider)


def _load_paraformer_zh(payload: Any) -> ParaformerZhConfig:
    if not isinstance(payload, dict):
        payload = {}
    return ParaformerZhConfig(
        model_dir=Path(str(payload.get("model_dir") or DEFAULT_PARAFORMER_MODEL_DIR)),
        device=str(payload.get("device") or "mps"),
    )


def _load_telemetry(payload: Any) -> TelemetryConfig:
    if not isinstance(payload, dict):
        payload = {}
    return TelemetryConfig(
        include_forced_units=bool(payload.get("include_forced_units", True)),
        include_asr_tokens=bool(payload.get("include_asr_tokens", True)),
        include_sentence_comparison=bool(payload.get("include_sentence_comparison", True)),
    )
