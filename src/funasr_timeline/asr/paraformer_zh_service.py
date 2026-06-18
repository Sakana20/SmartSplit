from __future__ import annotations

from pathlib import Path
from typing import Any

from funasr_timeline.asr.base import (
    AsrInfo,
    AsrService,
    AsrToken,
    AudioInfo,
    WordTimeline,
)

DEFAULT_PARAFORMER_MODEL_DIR = Path("/Users/sakana/PyEnv/paraformer")


class ParaformerZhAsrService(AsrService):
    provider = "paraformer-zh"

    def __init__(
        self,
        model_dir: Path = DEFAULT_PARAFORMER_MODEL_DIR,
        device: str = "mps",
        disable_update: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.disable_update = disable_update
        self._model: Any | None = None

    def transcribe(self, audio_path: Path) -> WordTimeline:
        model = self._load_model()
        results = model.generate(input=str(audio_path))
        result = _first_result(results)
        return paraformer_zh_result_to_timeline(
            result=result,
            audio_path=audio_path,
            model_dir=self.model_dir,
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Paraformer 模型目录不存在：{self.model_dir}")

        try:
            from funasr import AutoModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("缺少 funasr 运行时依赖，请先执行 `uv sync`。") from exc

        self._model = AutoModel(
            model=str(self.model_dir),
            device=self.device,
            disable_update=self.disable_update,
        )
        return self._model


def paraformer_zh_result_to_timeline(
    result: dict[str, Any],
    audio_path: Path,
    model_dir: Path,
) -> WordTimeline:
    text = str(result.get("text", ""))
    timestamps = _parse_timestamps(result.get("timestamp"))
    token_chars = _chars_for_timestamps(text=text, timestamp_count=len(timestamps))

    tokens = [
        AsrToken(
            index=index,
            text=char,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=None,
            source="paraformer-zh",
        )
        for index, (char, (start_ms, end_ms)) in enumerate(
            zip(token_chars, timestamps, strict=True)
        )
    ]

    return WordTimeline(
        audio=AudioInfo(
            path=str(audio_path),
            format=audio_path.suffix.lstrip(".").lower(),
            duration_ms=timestamps[-1][1] if timestamps else None,
        ),
        asr=AsrInfo(
            provider="paraformer-zh",
            model=f"paraformer-zh:{model_dir}",
            text=text,
        ),
        tokens=tokens,
    )


def _first_result(results: Any) -> dict[str, Any]:
    first = results[0] if isinstance(results, list) and results else results

    if not isinstance(first, dict):
        raise ValueError(f"FunASR 返回结构无法解析：{type(first).__name__}")
    return first


def _parse_timestamps(raw_timestamps: Any) -> list[tuple[int, int]]:
    if raw_timestamps is None:
        raise ValueError("FunASR 结果缺少 timestamp 字段，无法生成字符级时间轴。")
    if not isinstance(raw_timestamps, list):
        raise ValueError("FunASR timestamp 字段必须是列表。")

    timestamps: list[tuple[int, int]] = []
    for index, item in enumerate(raw_timestamps):
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not isinstance(item[0], int | float)
            or not isinstance(item[1], int | float)
        ):
            raise ValueError(f"FunASR timestamp[{index}] 不是有效的 [start_ms, end_ms]。")
        start_ms = int(item[0])
        end_ms = int(item[1])
        if end_ms < start_ms:
            raise ValueError(f"FunASR timestamp[{index}] 结束时间早于开始时间。")
        timestamps.append((start_ms, end_ms))
    return timestamps


def _chars_for_timestamps(text: str, timestamp_count: int) -> list[str]:
    non_space_chars = [char for char in text if not char.isspace()]
    if len(non_space_chars) == timestamp_count:
        return non_space_chars

    non_space_non_punctuation = [
        char for char in non_space_chars if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    ]
    if len(non_space_non_punctuation) == timestamp_count:
        return non_space_non_punctuation

    raise ValueError(
        "FunASR text 与 timestamp 数量无法对应："
        f"text_chars={len(non_space_chars)}, "
        f"text_chars_without_punctuation={len(non_space_non_punctuation)}, "
        f"timestamps={timestamp_count}"
    )
