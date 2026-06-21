from __future__ import annotations

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

DEFAULT_QWEN3_FORCED_ALIGNER_MODEL_DIR = Path("/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B")


class Qwen3ForcedAlignmentService(ForcedAlignmentService):
    provider = "qwen3-forced"

    def __init__(
        self,
        model_dir: Path = DEFAULT_QWEN3_FORCED_ALIGNER_MODEL_DIR,
        device_map: str = "mps",
        dtype: str = "bfloat16",
        max_audio_seconds: int = 300,
    ) -> None:
        self.model_dir = model_dir
        self.device_map = device_map
        self.dtype = dtype
        self.max_audio_seconds = max_audio_seconds
        self._model: Any | None = None

    def align(
        self,
        audio_path: Path,
        text: str,
        language: str,
    ) -> ForcedAlignmentResult:
        logger.debug(
            "开始 Qwen3 forced alignment：audio={} model={} device_map={} dtype={}",
            audio_path,
            self.model_dir,
            self.device_map,
            self.dtype,
        )

        model = self._load_model()

        raw_results = model.align(
            audio=str(audio_path),
            text=text,
            language=language,
        )

        logger.debug(
            "Qwen3 align 返回类型={} 长度={}",
            type(raw_results).__name__,
            len(raw_results) if hasattr(raw_results, "__len__") else "unknown",
        )

        raw_units = _extract_alignment_units(raw_results)

        logger.debug(
            "Qwen3 alignment units={}",
            len(raw_units),
        )

        units = [_convert_unit(item, index) for index, item in enumerate(raw_units)]

        normalized_text = normalize_text(text).text
        forced_normalized_text = "".join(unit.normalized_text for unit in units)

        duration_ms = max(
            (unit.end_ms for unit in units),
            default=None,
        )

        return ForcedAlignmentResult(
            audio=AudioInfo(
                path=str(audio_path),
                format=audio_path.suffix.lstrip(".").lower(),
                duration_ms=duration_ms,
            ),
            aligner=ForcedAlignmentInfo(
                provider=self.provider,
                model=str(self.model_dir),
                device_map=self.device_map,
                dtype=self.dtype,
                language=language,
            ),
            input_text=text,
            normalized_text=normalized_text,
            forced_normalized_text=forced_normalized_text,
            normalized_text_match=(normalized_text == forced_normalized_text),
            units=units,
            diagnostics={
                "max_audio_seconds": self.max_audio_seconds,
                "unit_count": len(units),
            },
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Qwen3 ForcedAligner 模型目录不存在：{self.model_dir}")

        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("缺少 qwen-asr/torch 运行时依赖，请先执行 `uv sync`。") from exc

        dtype = _torch_dtype(torch, self.dtype)

        logger.debug(
            "加载 Qwen3 ForcedAligner：model_dir={} device_map={} dtype={}",
            self.model_dir,
            self.device_map,
            self.dtype,
        )

        self._model = Qwen3ForcedAligner.from_pretrained(
            str(self.model_dir),
            dtype=dtype,
            device_map=self.device_map,
        )

        return self._model


def _extract_alignment_units(raw_results: Any) -> list[Any]:
    """
    官方返回:

    List[ForcedAlignResult]

    单条音频:
        results[0]

    ForcedAlignResult 本身可迭代:
        for item in results[0]:
            ...

    官方示例:
        results[0][0].text
    """

    if raw_results is None:
        return []

    if not isinstance(raw_results, list):
        logger.warning(
            "Qwen3 align 返回非 list 类型: {}",
            type(raw_results).__name__,
        )
        return list(raw_results)

    if len(raw_results) == 0:
        return []

    if len(raw_results) > 1:
        logger.warning(
            "Qwen3 align 返回 {} 个结果，仅使用第一个",
            len(raw_results),
        )

    first_result = raw_results[0]

    try:
        units = list(first_result)
    except Exception as exc:
        logger.exception(
            "无法遍历 ForcedAlignResult: {}",
            exc,
        )
        raise

    if units:
        first = units[0]
        logger.debug(
            "首个 alignment item: text='{}' start={} end={}",
            getattr(first, "text", ""),
            getattr(first, "start_time", None),
            getattr(first, "end_time", None),
        )

    return units


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    supported = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }

    try:
        return supported[dtype]
    except KeyError as exc:
        raise ValueError(f"不支持的 Qwen3 forced aligner dtype：{dtype}") from exc


def _convert_unit(
    item: Any,
    index: int,
) -> ForcedAlignmentUnit:
    text = str(getattr(item, "text", ""))

    return ForcedAlignmentUnit(
        index=index,
        text=text,
        start_ms=_seconds_to_ms(item.start_time),
        end_ms=_seconds_to_ms(item.end_time),
        normalized_text=normalize_text(text).text,
    )


def _seconds_to_ms(value: Any) -> int:
    return int(round(float(value) * 1000))
