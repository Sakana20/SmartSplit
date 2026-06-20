from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

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
        logger.debug("开始 paraformer-zh 识别：audio={} device={}", audio_path, self.device)
        model = self._load_model()
        results = model.generate(input=str(audio_path))
        result = _first_result(results)
        timeline = paraformer_zh_result_to_timeline(
            result=result,
            audio_path=audio_path,
            model_dir=self.model_dir,
        )
        logger.debug(
            "paraformer-zh 识别完成：tokens={} duration_ms={}",
            len(timeline.tokens),
            timeline.audio.duration_ms,
        )
        return timeline

    def _load_model(self) -> Any:
        if self._model is not None:
            logger.debug("复用已加载 paraformer-zh 模型：{}", self.model_dir)
            return self._model
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Paraformer 模型目录不存在：{self.model_dir}")

        try:
            from funasr import AutoModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("缺少 funasr 运行时依赖，请先执行 `uv sync`。") from exc

        logger.debug(
            "加载 paraformer-zh 模型：model_dir={} device={} disable_update={}",
            self.model_dir,
            self.device,
            self.disable_update,
        )
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
    # FunASR 的 timestamp 与文本字符数量可能因为空白或标点不同步，这里只接受可解释的映射。
    token_texts = _tokens_for_timestamps(text=text, timestamp_count=len(timestamps))
    logger.debug(
        "解析 paraformer-zh 输出：text_chars={} timestamps={} tokens={}",
        len(text),
        len(timestamps),
        len(token_texts),
    )

    tokens = [
        AsrToken(
            index=index,
            text=token_text,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=None,
            source="paraformer-zh",
        )
        for index, (token_text, (start_ms, end_ms)) in enumerate(
            zip(token_texts, timestamps, strict=True)
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


def _tokens_for_timestamps(text: str, timestamp_count: int) -> list[str]:
    non_space_chars = [char for char in text if not char.isspace()]
    if len(non_space_chars) == timestamp_count:
        return non_space_chars

    non_space_non_punctuation = [
        char for char in non_space_chars if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    ]
    if len(non_space_non_punctuation) == timestamp_count:
        return non_space_non_punctuation

    merged_tokens = _merge_ascii_runs_to_timestamp_count(
        chars=non_space_non_punctuation,
        timestamp_count=timestamp_count,
    )
    if merged_tokens is not None:
        logger.debug(
            "FunASR timestamp 少于文本字符，已合并 ASCII 连续片段："
            "chars={} timestamps={} merged_tokens={}",
            len(non_space_non_punctuation),
            timestamp_count,
            len(merged_tokens),
        )
        return merged_tokens

    merge_candidates = _ascii_run_diagnostics(non_space_non_punctuation)
    raise ValueError(
        "FunASR text 与 timestamp 数量无法对应："
        f"text_chars={len(non_space_chars)}, "
        f"text_chars_without_punctuation={len(non_space_non_punctuation)}, "
        f"timestamps={timestamp_count}, "
        f"ascii_run_candidates={merge_candidates[:20]}"
    )


def _merge_ascii_runs_to_timestamp_count(
    chars: list[str],
    timestamp_count: int,
) -> list[str] | None:
    """Merge small ASCII runs when FunASR emits one timestamp for multiple chars."""

    deficit = len(chars) - timestamp_count
    if deficit <= 0 or deficit > 20:
        return None

    tokens: list[str] = []
    cursor = 0
    remaining_deficit = deficit
    while cursor < len(chars):
        char = chars[cursor]
        if not _is_ascii_alnum(char):
            tokens.append(char)
            cursor += 1
            continue

        run_start = cursor
        while cursor < len(chars) and _is_ascii_alnum(chars[cursor]):
            cursor += 1
        run = chars[run_start:cursor]
        if len(run) <= 1 or remaining_deficit <= 0:
            tokens.extend(run)
            continue

        merge_size = min(len(run), remaining_deficit + 1)
        tokens.append("".join(run[:merge_size]))
        tokens.extend(run[merge_size:])
        remaining_deficit -= merge_size - 1

    if remaining_deficit != 0 or len(tokens) != timestamp_count:
        return None
    return tokens


def _ascii_run_diagnostics(chars: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(chars):
        if not _is_ascii_alnum(chars[cursor]):
            cursor += 1
            continue

        start = cursor
        while cursor < len(chars) and _is_ascii_alnum(chars[cursor]):
            cursor += 1
        if cursor - start <= 1:
            continue
        runs.append(
            {
                "start": start,
                "end": cursor,
                "text": "".join(chars[start:cursor]),
                "max_reduction": cursor - start - 1,
            }
        )
    return runs


def _is_ascii_alnum(char: str) -> bool:
    return char.isascii() and char.isalnum()
