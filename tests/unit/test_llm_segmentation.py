from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from funasr_timeline.segmentation import NO_SPLIT_END, NO_SPLIT_START
from funasr_timeline.segmentation.llm import (
    LlmSegmentationConfig,
    LlmSentenceSegmenter,
    load_llm_segmentation_config,
)


class _FakeResponse:
    def __init__(
        self, content: str | None = None, payload: dict[str, object] | None = None
    ) -> None:
        self.content = content
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        if self.payload is not None:
            return self.payload
        return {"choices": [{"message": {"content": self.content}}]}


class _FakeAsyncClient:
    requests: list[dict[str, Any]] = []
    response_content = ""
    response_contents: list[str] = []
    response_payload: dict[str, object] | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.requests.append({"url": url, **kwargs})
        content = self.response_contents.pop(0) if self.response_contents else self.response_content
        return _FakeResponse(content, self.response_payload)


def test_llm_segmenter_splits_multiple_blocks_in_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_contents = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
第一段。
第二段。
<<<BLOCK_SEPARATOR>>>
第三段，
继续。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment("第一段。第二段。\n第三段，继续。")

    assert [segment.text for segment in result.segments] == ["第一段", "第二段", "第三段", "继续"]
    assert [segment.paragraph_index for segment in result.segments] == [0, 0, 1, 1]
    assert len(_FakeAsyncClient.requests) == 1
    assert _FakeAsyncClient.requests[0]["url"] == "https://example.test/v1/chat/completions"
    payload = _FakeAsyncClient.requests[0]["json"]
    assert payload["model"] == "test-model"
    assert payload["enable_thinking"] is False
    prompt = payload["messages"][0]["content"]
    assert "第一段。第二段。" in prompt
    assert "第三段，继续。" in prompt
    assert "不要输出 XML" in prompt
    assert "不要输出 JSON" in prompt
    assert "只允许插入分句边界" in prompt
    assert "标点符号也属于原文字符" in prompt
    assert "所有输出行直接拼接后，必须与对应 input_block 原文完全一致" in prompt
    assert "常规口播字幕优先控制在 4 到 12 个中文字符左右" in prompt
    assert "中文字符必须不超过 14 个" in prompt
    assert "英文、数字、标点符号不计入长度" in prompt
    assert "特别" in prompt
    assert "再比如" in prompt
    assert "不要为了字幕观感在数字、英文、单位之间添加空格" in prompt


def test_llm_segmenter_retries_when_segment_exceeds_hard_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = [
        "天气热的时候最怕空气闷闷的，这款便携手持风扇轻松带来凉爽体验。",
        "天气热的时候\n最怕空气闷闷的，\n这款便携手持风扇\n轻松带来凉爽体验。",
    ]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment(
        "天气热的时候最怕空气闷闷的，这款便携手持风扇轻松带来凉爽体验。"
    )

    assert [segment.text for segment in result.segments] == [
        "天气热的时候",
        "最怕空气闷闷的",
        "这款便携手持风扇",
        "轻松带来凉爽体验",
    ]
    assert len(_FakeAsyncClient.requests) == 2
    retry_prompt = _FakeAsyncClient.requests[1]["json"]["messages"][0]["content"]
    assert "segment_too_long" in retry_prompt
    assert "按意群边界继续拆分" in retry_prompt


def test_llm_segmenter_length_check_counts_only_chinese_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = []
    _FakeAsyncClient.response_content = """
iPhone15手机壳、500ml矿泉水、2kg大米。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment("iPhone15手机壳、500ml矿泉水、2kg大米。")

    assert [segment.text for segment in result.segments] == ["iPhone15手机壳、500ml矿泉水、2kg大米"]
    assert len(_FakeAsyncClient.requests) == 1


def test_llm_segmenter_keeps_protected_block_as_single_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
开头。
<<<BLOCK_SEPARATOR>>>
结尾。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)
    text = f"开头。{NO_SPLIT_START}这里。不要切！{NO_SPLIT_END}结尾。"

    result = LlmSentenceSegmenter(_config()).segment(text)

    assert [segment.text for segment in result.segments] == ["开头", "这里。不要切！", "结尾"]
    assert [segment.boundary for segment in result.segments] == ["llm", "protected", "llm"]


def test_llm_segmenter_accepts_plaintext_lines_with_boundary_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
第一段。
第二段。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment("第一段。第二段。")

    assert [segment.text for segment in result.segments] == ["第一段", "第二段"]


def test_llm_segmenter_removes_boundary_punctuation_after_exact_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
满减后只要9.9元，
还能叠加最高12元无门槛红包。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment(
        "满减后只要9.9元，还能叠加最高12元无门槛红包。"
    )

    assert [segment.text for segment in result.segments] == [
        "满减后只要9.9元",
        "还能叠加最高12元无门槛红包",
    ]


def test_llm_segmenter_raises_when_llm_adds_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
满减后只要 9.9 元，
还能叠加最高12元无门槛红包。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="重试后仍未通过校验"):
        LlmSentenceSegmenter(_config()).segment("满减后只要9.9元，还能叠加最高12元无门槛红包。")

    payload = _FakeAsyncClient.requests[-1]["json"]
    prompt = payload["messages"][0]["content"]
    assert "不要为了字幕观感在数字、英文、单位之间添加空格" in prompt
    assert "9.9元" in prompt
    assert "iPhone15" in prompt


def test_llm_segmenter_raises_when_output_does_not_cover_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
第一段。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="重试后仍未通过校验"):
        LlmSentenceSegmenter(_config()).segment("第一段。第二段。")


def test_llm_segmenter_writes_validation_diagnostics_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    diagnostics_path = tmp_path / "llm_segmentation_diagnostics.json"
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
如果你今天刚好要买纸巾。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="重试后仍未通过校验"):
        LlmSentenceSegmenter(_config(diagnostics_path=diagnostics_path)).segment(
            "长句测试：如果你今天刚好要买纸巾。"
        )

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["status"] == "failed"
    assert diagnostics["parsed"] == {}
    assert diagnostics["validation"]["reason"] == "retry_exhausted"
    last_error = diagnostics["validation"]["last_error"]
    assert last_error["diagnostics"]["failed_block_id"] == "block-0"
    assert last_error["diagnostics"]["reason"] == "text_not_equal"
    assert last_error["previous_output"] == _FakeAsyncClient.response_content.strip()


def test_llm_segmenter_reports_empty_content_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = {
        "choices": [{"finish_reason": "length", "message": {"content": ""}}]
    }
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="finish_reason='length'.*message_keys=\\['content'\\]"):
        LlmSentenceSegmenter(_config()).segment("第一段。")


def test_load_llm_segmentation_config_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.toml"
    config_path.write_text(
        """
[llm]
base_url = "https://api.example.test/v1"
model = "Qwen/Qwen3.5-9B"
api_key_env = "TEST_LLM_KEY"
timeout_seconds = 12
temperature = 0
max_tokens = 1234
enable_thinking = false
diagnostics_path = "tmp/llm.json"
""",
        encoding="utf-8",
    )

    config = load_llm_segmentation_config(config_path)

    assert config.base_url == "https://api.example.test/v1"
    assert config.model == "Qwen/Qwen3.5-9B"
    assert config.api_key_env == "TEST_LLM_KEY"
    assert config.timeout_seconds == 12
    assert config.max_tokens == 1234
    assert config.enable_thinking is False
    assert config.diagnostics_path == Path("tmp/llm.json")


def test_load_llm_segmentation_config_rejects_non_bool_enable_thinking(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "llm.toml"
    config_path.write_text(
        """
[llm]
base_url = "https://api.example.test/v1"
model = "Qwen/Qwen3.5-9B"
api_key_env = "TEST_LLM_KEY"
enable_thinking = "false"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="llm.enable_thinking"):
        load_llm_segmentation_config(config_path)


def _config(diagnostics_path: Path | None = None) -> LlmSegmentationConfig:
    return LlmSegmentationConfig(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_LLM_KEY",
        api_key="test-key",
        diagnostics_path=diagnostics_path,
    )
