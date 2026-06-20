from __future__ import annotations

from funasr_timeline.forced_alignment.base import ForcedAlignmentService
from funasr_timeline.forced_alignment.config import Qwen3ForcedConfig
from funasr_timeline.forced_alignment.mock_service import MockForcedAlignmentService
from funasr_timeline.forced_alignment.qwen3_service import Qwen3ForcedAlignmentService


def create_forced_alignment_service(config: Qwen3ForcedConfig) -> ForcedAlignmentService:
    if config.provider == "mock":
        if config.units_path is None:
            raise ValueError("qwen3_forced.provider=mock 需要配置 units_path")
        return MockForcedAlignmentService(config.units_path)
    if config.provider == "qwen3-forced":
        return Qwen3ForcedAlignmentService(
            model_dir=config.model_dir,
            device_map=config.device_map,
            dtype=config.dtype,
            max_audio_seconds=config.max_audio_seconds,
        )
    raise ValueError(f"不支持的 forced alignment provider：{config.provider}")
