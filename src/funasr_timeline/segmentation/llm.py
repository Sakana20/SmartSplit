from __future__ import annotations

import asyncio
import json
import os
import re
import tomllib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Template
from loguru import logger

from funasr_timeline.segmentation.base import SegmentationResult, SentenceSegment
from funasr_timeline.segmentation.protection import (
    TextBlock,
    append_protected_segment,
    split_text_blocks,
)

DEFAULT_LLM_CONFIG_PATH = Path("configs/llm-siliconflow.toml")
_BLOCK_SEPARATOR = "<<<BLOCK_SEPARATOR>>>"
_LLM_SEGMENT_TARGET_MAX_CONTENT_CHARS = 12
_LLM_SEGMENT_HARD_MAX_CONTENT_CHARS = 14


@dataclass(frozen=True, slots=True)
class LlmSegmentationConfig:
    base_url: str
    model: str
    api_key_env: str
    api_key: str | None = None
    timeout_seconds: float = 90.0
    temperature: float = 0.0
    max_tokens: int = 4096
    enable_thinking: bool = False
    diagnostics_path: Path | None = None
    max_retries: int = 3

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ValueError(f"缺少 LLM API Key：请在环境变量 {self.api_key_env} 中配置")
        return value


@dataclass(frozen=True, slots=True)
class _PromptBlock:
    block_id: str
    paragraph_index: int
    text: str


@dataclass(frozen=True, slots=True)
class _FewShotExample:
    input_text: str
    output_text: str


@dataclass(frozen=True, slots=True)
class _LlmAttempt:
    attempt: int
    prompt: str
    content: str
    parsed: dict[str, tuple[str, ...]]
    error: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _LlmSegmentationResponse:
    content: str
    payload: dict[str, Any]
    parsed: dict[str, tuple[str, ...]]
    finish_reason: object
    attempts: tuple[_LlmAttempt, ...] = ()


class LlmSegmentLocationError(ValueError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class LlmSentenceSegmenter:
    name = "llm"

    def __init__(self, config: LlmSegmentationConfig) -> None:
        self.config = config

    def segment(self, text: str) -> SegmentationResult:
        prepared_text, blocks = split_text_blocks(text)

        prompt_blocks: list[_PromptBlock] = []
        block_id_by_index: dict[int, str] = {}

        for index, block in enumerate(blocks):
            if block.protected or not block.text.strip():
                continue

            block_id = f"block-{len(prompt_blocks)}"
            block_id_by_index[index] = block_id
            prompt_blocks.append(
                _PromptBlock(
                    block_id=block_id,
                    paragraph_index=block.paragraph_index,
                    text=block.text,
                )
            )

        logger.debug(
            "LLM 分句开始：blocks={} prompt_blocks={} chars={}",
            len(blocks),
            len(prompt_blocks),
            len(prepared_text),
        )

        try:
            response = (
                _run_async(self._segment_blocks(prompt_blocks))
                if prompt_blocks
                else _LlmSegmentationResponse(content="", payload={}, parsed={}, finish_reason=None)
            )
        except LlmSegmentLocationError as error:
            self._write_diagnostics(
                status="failed",
                prompt_blocks=prompt_blocks,
                response=_LlmSegmentationResponse(
                    content="",
                    payload={},
                    parsed={},
                    finish_reason=None,
                ),
                validation=error.diagnostics,
            )
            raise

        self._write_diagnostics(
            status="parsed",
            prompt_blocks=prompt_blocks,
            response=response,
            validation=None,
        )

        segments: list[SentenceSegment] = []

        try:
            for index, block in enumerate(blocks):
                if block.protected:
                    append_protected_segment(segments, block)
                    continue

                if not block.text.strip():
                    continue

                block_id = block_id_by_index[index]

                for char_start, char_end, segment_text in _locate_segments_in_block(
                    block=block,
                    block_id=block_id,
                    segment_texts=response.parsed.get(block_id, ()),
                ):
                    segments.append(
                        SentenceSegment(
                            index=len(segments),
                            text=segment_text,
                            paragraph_index=block.paragraph_index,
                            char_start=char_start,
                            char_end=char_end,
                            boundary="llm",
                        )
                    )

        except LlmSegmentLocationError as error:
            self._write_diagnostics(
                status="failed",
                prompt_blocks=prompt_blocks,
                response=response,
                validation=error.diagnostics,
            )
            raise

        logger.debug("LLM 分句完成：segments={}", len(segments))

        self._write_diagnostics(
            status="validated",
            prompt_blocks=prompt_blocks,
            response=response,
            validation={"segment_count": len(segments)},
        )

        return SegmentationResult(text=prepared_text, segments=segments)

    async def _segment_blocks(self, blocks: list[_PromptBlock]) -> _LlmSegmentationResponse:
        endpoint = _chat_completions_url(self.config.base_url)
        headers = {
            "Authorization": f"Bearer {self.config.resolved_api_key}",
            "Content-Type": "application/json",
        }

        last_error: dict[str, Any] | None = None
        last_content = ""
        last_payload: dict[str, Any] = {}
        last_finish_reason: object = None
        attempts: list[_LlmAttempt] = []

        max_attempts = max(1, self.config.max_retries + 1)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout_seconds)) as client:
            for attempt in range(1, max_attempts + 1):
                prompt = _render_prompt(
                    blocks=blocks,
                    previous_error=last_error,
                    previous_output=None,
                )

                logger.debug("LLM 分句第 {} 次输入 prompt：\n{}", attempt, prompt)

                payload = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "enable_thinking": self.config.enable_thinking,
                }

                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()

                last_payload = response.json()
                last_finish_reason = _first_choice_finish_reason(last_payload)
                last_content = _extract_message_content(last_payload)

                logger.debug("LLM 分句第 {} 次原始输出：\n{}", attempt, last_content)

                parsed: dict[str, tuple[str, ...]] = {}

                try:
                    parsed = _parse_plaintext_segments(last_content, blocks)

                    logger.debug(
                        "LLM 分句第 {} 次解析结果：\n{}",
                        attempt,
                        json.dumps(
                            {key: list(value) for key, value in parsed.items()},
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )

                    _validate_parsed_segments(blocks, parsed)

                    logger.debug("LLM 分句第 {} 次校验通过", attempt)

                    attempts.append(
                        _LlmAttempt(
                            attempt=attempt,
                            prompt=prompt,
                            content=last_content,
                            parsed=parsed,
                            error=None,
                        )
                    )

                    return _LlmSegmentationResponse(
                        content=last_content,
                        payload=last_payload,
                        parsed=parsed,
                        finish_reason=last_finish_reason,
                        attempts=tuple(attempts),
                    )

                except Exception as exc:
                    last_error = _error_to_retry_payload(
                        exc=exc,
                        blocks=blocks,
                        previous_output=last_content,
                    )

                    logger.warning(
                        "LLM 分句第 {} 次失败：\n{}",
                        attempt,
                        json.dumps(last_error, ensure_ascii=False, indent=2),
                    )

                    attempts.append(
                        _LlmAttempt(
                            attempt=attempt,
                            prompt=prompt,
                            content=last_content,
                            parsed=parsed,
                            error=last_error,
                        )
                    )

        raise LlmSegmentLocationError(
            "LLM 分句重试后仍未通过校验",
            {
                "reason": "retry_exhausted",
                "max_retries": self.config.max_retries,
                "last_error": last_error,
                "last_output": last_content,
            },
        )

    def with_diagnostics_path(self, diagnostics_path: Path) -> LlmSentenceSegmenter:
        return LlmSentenceSegmenter(
            LlmSegmentationConfig(
                base_url=self.config.base_url,
                model=self.config.model,
                api_key_env=self.config.api_key_env,
                api_key=self.config.api_key,
                timeout_seconds=self.config.timeout_seconds,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                enable_thinking=self.config.enable_thinking,
                diagnostics_path=diagnostics_path,
                max_retries=self.config.max_retries,
            )
        )

    def _write_diagnostics(
        self,
        *,
        status: str,
        prompt_blocks: list[_PromptBlock],
        response: _LlmSegmentationResponse,
        validation: dict[str, Any] | None,
    ) -> None:
        if self.config.diagnostics_path is None:
            return

        payload = {
            "status": status,
            "config": {
                "base_url": self.config.base_url,
                "model": self.config.model,
                "api_key_env": self.config.api_key_env,
                "timeout_seconds": self.config.timeout_seconds,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "enable_thinking": self.config.enable_thinking,
                "max_retries": self.config.max_retries,
                "segment_target_max_chinese_chars": (_LLM_SEGMENT_TARGET_MAX_CONTENT_CHARS),
                "segment_hard_max_chinese_chars": (_LLM_SEGMENT_HARD_MAX_CONTENT_CHARS),
            },
            "request": {
                "block_count": len(prompt_blocks),
                "blocks": [
                    {
                        "id": block.block_id,
                        "paragraph_index": block.paragraph_index,
                        "text": block.text,
                    }
                    for block in prompt_blocks
                ],
            },
            "response": {
                "finish_reason": response.finish_reason,
                "content": response.content,
            },
            "parsed": {key: list(value) for key, value in response.parsed.items()},
            "attempts": [
                {
                    "attempt": attempt.attempt,
                    "prompt": attempt.prompt,
                    "content": attempt.content,
                    "parsed": {key: list(value) for key, value in attempt.parsed.items()},
                    "error": attempt.error,
                }
                for attempt in response.attempts
            ],
            "validation": validation,
        }

        self.config.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.diagnostics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_llm_segmentation_config(path: Path) -> LlmSegmentationConfig:
    if not path.exists():
        raise FileNotFoundError(f"LLM 配置文件不存在：{path}")

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    llm = payload.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("LLM 配置文件必须包含 [llm] 表")

    base_url = _required_str(llm, "base_url")
    model = _required_str(llm, "model")
    api_key_env = str(llm.get("api_key_env") or "FUNASR_TIMELINE_LLM_API_KEY")

    api_key = llm.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError("llm.api_key 必须是字符串")

    return LlmSegmentationConfig(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        api_key=api_key,
        timeout_seconds=float(llm.get("timeout_seconds", 90.0)),
        temperature=float(llm.get("temperature", 0.0)),
        max_tokens=int(llm.get("max_tokens", 4096)),
        enable_thinking=_optional_bool(llm.get("enable_thinking"), default=False),
        diagnostics_path=_optional_path(llm.get("diagnostics_path")),
        max_retries=int(llm.get("max_retries", 3)),
    )


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM 配置缺少字符串字段：llm.{key}")
    return value.strip()


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("llm.enable_thinking 必须是布尔值")


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return Path(value.strip())
    raise ValueError("llm.diagnostics_path 必须是非空字符串")


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("LLM 分句不能在已有 asyncio event loop 中同步调用")


def _chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM 响应缺少 choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("LLM 响应 choices[0] 格式错误")

    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM 响应缺少 message")

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    message_keys = sorted(str(key) for key in message)
    finish_reason = first.get("finish_reason")
    raise ValueError(
        "LLM 响应 message.content 为空："
        f"finish_reason={finish_reason!r} message_keys={message_keys}"
    )


def _first_choice_finish_reason(payload: dict[str, Any]) -> object:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def _parse_plaintext_segments(
    content: str,
    blocks: list[_PromptBlock],
) -> dict[str, tuple[str, ...]]:
    text = _strip_code_fence(content)

    chunks = [text] if len(blocks) == 1 else text.split(_BLOCK_SEPARATOR)

    if len(chunks) != len(blocks):
        raise LlmSegmentLocationError(
            "LLM 输出 block 数量不匹配",
            {
                "reason": "block_count_mismatch",
                "expected_count": len(blocks),
                "actual_count": len(chunks),
                "separator": _BLOCK_SEPARATOR,
                "content": content,
            },
        )

    parsed: dict[str, tuple[str, ...]] = {}

    for block, chunk in zip(blocks, chunks, strict=True):
        normalized_chunk = chunk.strip("\n")
        segments = tuple(
            cleaned for line in normalized_chunk.split("\n") if (cleaned := line.strip()) != ""
        )

        logger.debug(
            "LLM 分句 block={} 清洗后分句：{}",
            block.block_id,
            list(segments),
        )

        if not segments:
            raise LlmSegmentLocationError(
                f"LLM 输出空 block：{block.block_id}",
                {
                    "reason": "empty_block_output",
                    "failed_block_id": block.block_id,
                    "original_text": block.text,
                    "chunk": chunk,
                },
            )

        parsed[block.block_id] = segments

    return parsed


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:text|txt)?\s*(?P<body>.*?)\s*```", stripped, re.DOTALL)
    return match.group("body").strip("\n") if match else stripped


def _validate_parsed_segments(
    blocks: list[_PromptBlock],
    parsed: dict[str, tuple[str, ...]],
) -> None:
    expected_ids = {block.block_id for block in blocks}
    actual_ids = set(parsed)

    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)

    if missing_ids or extra_ids:
        raise LlmSegmentLocationError(
            "LLM 分句输出段落不匹配",
            {
                "reason": "block_id_mismatch",
                "missing_ids": missing_ids,
                "extra_ids": extra_ids,
            },
        )

    block_by_id = {block.block_id: block for block in blocks}

    for block_id, segment_texts in parsed.items():
        block = block_by_id[block_id]
        consumed_text = "".join(segment_texts)

        if consumed_text != block.text:
            raise LlmSegmentLocationError(
                f"LLM 分句结果未完整覆盖原文：block={block_id}",
                {
                    "reason": "text_not_equal",
                    "failed_block_id": block_id,
                    "original_text": block.text,
                    "consumed_text": consumed_text,
                    "missing_hint": _first_difference_hint(block.text, consumed_text),
                    "segments": list(segment_texts),
                },
            )

        for segment_index, segment_text in enumerate(segment_texts):
            content_text = _strip_segment_boundary_noise(segment_text)
            chinese_char_count = _count_chinese_chars(content_text)

            if chinese_char_count > _LLM_SEGMENT_HARD_MAX_CONTENT_CHARS:
                raise LlmSegmentLocationError(
                    f"LLM 分句结果超过长度上限：block={block_id} "
                    f"segment_index={segment_index} chinese_chars={chinese_char_count}",
                    {
                        "reason": "segment_too_long",
                        "failed_block_id": block_id,
                        "segment_index": segment_index,
                        "segment_text": segment_text,
                        "content_text": content_text,
                        "chinese_char_count": chinese_char_count,
                        "target_max_chinese_chars": (_LLM_SEGMENT_TARGET_MAX_CONTENT_CHARS),
                        "hard_max_chinese_chars": _LLM_SEGMENT_HARD_MAX_CONTENT_CHARS,
                        "repair_hint": (
                            "该分句过长，请在意群边界、自然停顿、并列枚举、"
                            "动作切换或软断点处继续拆分。"
                        ),
                        "segments": list(segment_texts),
                    },
                )


def _locate_segments_in_block(
    block: TextBlock,
    block_id: str,
    segment_texts: Iterable[str],
) -> list[tuple[int, int, str]]:
    segment_text_list = tuple(segment_texts)
    spans = _locate_relaxed_spans(
        original_text=block.text,
        block_id=block_id,
        segment_texts=segment_text_list,
    )

    located: list[tuple[int, int, str]] = []

    for start, end in spans:
        start, end = _trim_span_boundary_noise(block.text, start, end)

        if start >= end:
            continue

        segment_text = block.text[start:end]
        if not segment_text:
            continue

        located.append(
            (
                block.start + start,
                block.start + end,
                segment_text,
            )
        )

    return located


def _locate_relaxed_spans(
    *,
    original_text: str,
    block_id: str,
    segment_texts: tuple[str, ...],
) -> list[tuple[int, int]]:
    original_core = _content_core(original_text)
    consumed_core = _content_core("".join(segment_texts))

    if original_core != consumed_core:
        raise LlmSegmentLocationError(
            f"LLM 分句结果无法宽松定位：block={block_id}",
            {
                "reason": "normalized_content_not_equal_before_locate",
                "failed_block_id": block_id,
                "original_text": original_text,
                "consumed_text": "".join(segment_texts),
                "original_core": original_core,
                "consumed_core": consumed_core,
                "missing_hint": _first_difference_hint(original_core, consumed_core),
            },
        )

    spans: list[tuple[int, int]] = []
    cursor = 0
    pending_ignored_start: int | None = None

    for segment_text in segment_texts:
        target_core = _content_core(segment_text)

        if target_core == "":
            continue

        while cursor < len(original_text) and not _is_content_char(original_text[cursor]):
            if spans:
                start, _ = spans[-1]
                spans[-1] = (start, cursor + 1)
            else:
                pending_ignored_start = (
                    cursor if pending_ignored_start is None else pending_ignored_start
                )
            cursor += 1

        start = pending_ignored_start if pending_ignored_start is not None else cursor
        pending_ignored_start = None

        target_index = 0

        while cursor < len(original_text) and target_index < len(target_core):
            char = original_text[cursor]

            if not _is_content_char(char):
                cursor += 1
                continue

            normalized_char = _normalize_content_char(char)

            if normalized_char != target_core[target_index]:
                raise LlmSegmentLocationError(
                    f"LLM 分句结果顺序定位失败：block={block_id}",
                    {
                        "reason": "relaxed_segment_order_mismatch",
                        "failed_block_id": block_id,
                        "failed_segment": segment_text,
                        "segment_core": target_core,
                        "target_index": target_index,
                        "expected_char": target_core[target_index],
                        "actual_char": normalized_char,
                        "cursor": cursor,
                        "original_text": original_text,
                    },
                )

            target_index += 1
            cursor += 1

        if target_index != len(target_core):
            raise LlmSegmentLocationError(
                f"LLM 分句结果未能完整定位片段：block={block_id}",
                {
                    "reason": "relaxed_segment_not_fully_matched",
                    "failed_block_id": block_id,
                    "failed_segment": segment_text,
                    "segment_core": target_core,
                    "matched_count": target_index,
                    "cursor": cursor,
                    "original_text": original_text,
                },
            )

        spans.append((start, cursor))

    while cursor < len(original_text):
        if spans:
            start, _ = spans[-1]
            spans[-1] = (start, cursor + 1)
        cursor += 1

    return spans


def _strip_segment_boundary_noise(text: str) -> str:
    """去除 LLM 单个分句首尾的标点、分隔符、空格、换行等非内容字符。"""
    start = 0
    end = len(text)

    while start < end and not _is_content_char(text[start]):
        start += 1

    while end > start and not _is_content_char(text[end - 1]):
        end -= 1

    return text[start:end]


def _trim_span_boundary_noise(text: str, start: int, end: int) -> tuple[int, int]:
    """把定位到的原文 span 首尾非内容字符裁掉，保证输出分句不带边界标点/空白。"""
    start = max(0, start)
    end = min(len(text), end)

    while start < end and not _is_content_char(text[start]):
        start += 1

    while end > start and not _is_content_char(text[end - 1]):
        end -= 1

    return start, end


def _content_core(text: str) -> str:
    return "".join(_normalize_content_char(char) for char in text if _is_content_char(char))


def _count_chinese_chars(text: str) -> int:
    return sum(1 for char in text if _is_cjk_char(char))


def _is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_content_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N"}


def _normalize_content_char(char: str) -> str:
    return unicodedata.normalize("NFKC", char).casefold()


def _error_to_retry_payload(
    *,
    exc: Exception,
    blocks: list[_PromptBlock],
    previous_output: str,
) -> dict[str, Any]:
    if isinstance(exc, LlmSegmentLocationError):
        diagnostics = exc.diagnostics
    else:
        diagnostics = {
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    return {
        "message": str(exc),
        "diagnostics": diagnostics,
        "original_blocks": [
            {
                "id": block.block_id,
                "text": block.text,
            }
            for block in blocks
        ],
        "previous_output": previous_output,
        "repair_instruction": (
            "请重新输出完整结果。只需要表达分句换行。"
            "只允许插入分句边界，不允许新增、删除、替换或改写任何字符。"
            "标点符号也属于原文字符，必须保留在输出行中。"
            f"每个分句去掉边界标点后的中文字符必须不超过 "
            f"{_LLM_SEGMENT_HARD_MAX_CONTENT_CHARS} 个；"
            "英文、数字、标点符号不计入长度。过长时按意群边界继续拆分。"
            f"多个 block 之间必须使用 {_BLOCK_SEPARATOR} 单独一行分隔。"
        ),
    }


def _first_difference_hint(original_text: str, consumed_text: str) -> str:
    prefix_len = 0

    for original_char, consumed_char in zip(original_text, consumed_text, strict=False):
        if original_char != consumed_char:
            break
        prefix_len += 1

    if prefix_len < len(original_text):
        return original_text[prefix_len : prefix_len + 24]

    if prefix_len < len(consumed_text):
        return consumed_text[prefix_len : prefix_len + 24]

    return ""


def _render_prompt(
    *,
    blocks: list[_PromptBlock],
    previous_error: dict[str, Any] | None = None,
    previous_output: str | None = None,
) -> str:
    previous_error_json = (
        json.dumps(previous_error, ensure_ascii=False, indent=2)
        if previous_error is not None
        else None
    )

    return Template(_PROMPT_TEMPLATE, trim_blocks=True, lstrip_blocks=True).render(
        blocks=blocks,
        examples=_FEW_SHOT_EXAMPLES,
        protected_phrases=_PROTECTED_PHRASES,
        soft_split_markers=_SOFT_SPLIT_MARKERS,
        mixed_content_examples=_MIXED_CONTENT_EXAMPLES,
        previous_error=previous_error,
        previous_error_json=previous_error_json,
        previous_output=previous_output,
        block_separator=_BLOCK_SEPARATOR,
        target_max_chinese_chars=_LLM_SEGMENT_TARGET_MAX_CONTENT_CHARS,
        hard_max_chinese_chars=_LLM_SEGMENT_HARD_MAX_CONTENT_CHARS,
    )


_PROTECTED_PHRASES = (
    "淘宝闪购",
    "天猫超市",
    "饿了么",
    "最高12元无门槛红包",
    "无门槛红包",
    "视频下方链接",
    "下方链接",
    "官方补贴",
    "今日特价",
    "限时秒杀",
    "满减活动",
    "便携手持风扇",
)

_SOFT_SPLIT_MARKERS = (
    "特别",
    "直接",
    "真的",
    "比如",
    "再比如",
)

_MIXED_CONTENT_EXAMPLES = (
    "9.9元",
    "99.9%",
    "500ml",
    "2kg",
    "3.5L",
    "iPhone15",
    "USD 3.99",
    "No.1爆款",
)

_FEW_SHOT_EXAMPLES = (
    _FewShotExample(
        input_text="打开视频下方链接，先领最高12元无门槛红包，再看附近门店有没有你要的东西。",
        output_text=(
            "打开视频下方链接，\n先领最高12元无门槛红包，\n再看附近门店有没有你要的东西。"
        ),
    ),
    _FewShotExample(
        input_text="iPhone15手机壳、500ml矿泉水、2kg大米、3.5L洗衣液、99.9%除菌湿巾。",
        output_text=("iPhone15手机壳、\n500ml矿泉水、\n2kg大米、\n3.5L洗衣液、\n99.9%除菌湿巾。"),
    ),
    _FewShotExample(
        input_text=(
            "学生党上下学路上特别容易热得满头汗，包里放个便携手持风扇就方便多了。"
            "随拿随用，清凉一路陪伴。淘宝闪购最高12元无门槛红包正在发放，"
            "点击视频下方链接看看。"
        ),
        output_text=(
            "学生党上下学路上\n"
            "特别容易\n"
            "热得满头汗，\n"
            "包里放个便携手持风扇\n"
            "就方便多了。\n"
            "随拿随用，\n"
            "清凉一路陪伴。\n"
            "淘宝闪购最高12元无门槛红包\n"
            "正在发放，\n"
            "点击视频下方链接看看。"
        ),
    ),
)

_PROMPT_TEMPLATE = """你是一个中文短视频字幕分句助手。

你的任务：
把输入文本切分成适合短视频口播节奏的字幕句。

输出方式：
- 不要输出 XML。
- 不要输出 JSON。
- 不要输出解释。
- 只输出“在原文中插入换行后的文本”。
- 每一行就是一个字幕分句。
- 只允许插入分句边界，不允许新增、删除、替换、改写任何字符。
- 标点符号也属于原文字符，必须保留在输出行中，用于后续完整性校对。
- 每个 block 内所有输出行直接拼接后，必须与对应 input_block 原文完全一致。
- 不要为了字幕观感在数字、英文、单位之间添加空格。
- 不要遗漏中文、数字、英文、标点或原文中的空格。

多 block 规则：
- 每个 input_block 必须按输入顺序输出。
- 不要输出 block id。
- 不要输出标题。
- 不要跨 block 合并。
- 多个 block 之间必须使用这一行分隔：
{{ block_separator }}

短视频字幕偏好：
- 优先按口播自然停顿切分。
- 常规口播字幕优先控制在 4 到 {{ target_max_chinese_chars }} 个中文字符左右。
- 每个分句去掉首尾边界标点后，中文字符必须不超过 {{ hard_max_chinese_chars }} 个。
- 英文、数字、标点符号不计入长度；只计算中文文字部分长度。
- 如果某个分句超过 {{ hard_max_chinese_chars }} 个中文字符，必须按意群边界继续拆分。
- 品牌、商品名、权益短语、链接引导、数字单位、英文型号可以适当更长。
- 这些较长片段仍需遵守 {{ hard_max_chinese_chars }} 个中文字符上限，不要为了字数拆坏关键词。
- 可以把过长分句按“主语/动作/结果”“条件/动作”“商品/卖点”“权益/CTA”等意群边界继续拆开。
- 优先在逗号、顿号、句号、分号、语义转折、并列枚举、动作切换处断开。
- 枚举商品、规格、场景时，可以按单个枚举项拆分。
- 商品名、品牌名、红包权益、链接引导、数字单位、英文型号尽量不要拆开。
- CTA 类表达如“点击视频下方链接”“打开下方链接”“直接看看”尽量保持完整。
- 如果不确定，宁可保持更长一句，也不要冒险改字、漏字或拆坏关键词。

可参考的口播软断点：
{% for marker in soft_split_markers -%}
- {{ marker }}
{% endfor %}

保护短语尽量不要拆开：
{% for phrase in protected_phrases -%}
- {{ phrase }}
{% endfor %}

混合内容内部尽量不要拆开：
{% for item in mixed_content_examples -%}
- {{ item }}
{% endfor %}

Few-shot 示例：
{% for example in examples %}

输入：
{{ example.input_text }}

输出：
{{ example.output_text }}
{% endfor %}

{% if previous_error %}
上一次输出没有通过程序校验，请重新输出。

上一次错误：
{{ previous_error_json }}

修复要求：
- 不要解释。
- 只重新选择换行位置。
- 只允许插入分句边界，不允许新增、删除、替换、改写任何字符。
- 标点符号也属于原文字符，必须保留在输出行中。
- 每个 block 内所有输出行直接拼接后，必须与对应 input_block 原文完全一致。
- 不要为了字幕观感在数字、英文、单位之间添加空格。
- 每个分句去掉首尾边界标点后，中文字符必须不超过 {{ hard_max_chinese_chars }} 个。
- 英文、数字、标点符号不计入长度。
- 过长时按意群边界继续拆分。
- 如果有多个 block，必须用 {{ block_separator }} 单独一行分隔。
{% endif %}

现在处理下面的输入。只输出加换行后的文本。

{% for block in blocks %}
<input_block id="{{ block.block_id }}">
{{ block.text }}
</input_block>
{% endfor %}
"""
