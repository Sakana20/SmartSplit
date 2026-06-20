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
        return _FakeResponse(self.response_content, self.response_payload)


def test_llm_segmenter_splits_multiple_blocks_in_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
<segmentation>
  <block id="block-0">
    <segment>第一段。</segment>
    <segment>第二段。</segment>
  </block>
  <block id="block-1">
    <segment>第三段，</segment>
    <segment>继续。</segment>
  </block>
</segmentation>
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
    assert "只允许插入分句边界" in prompt
    assert "标点符号也属于原文字符" in prompt
    assert "直接拼接后，必须与对应 input_block 原文完全一致" in prompt
    assert '输出 <block id=""> 必须与输入 <input_block id=""> 完全一致' in prompt


def test_llm_segmenter_keeps_protected_block_as_single_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
<segmentation>
  <block id="block-0"><segment>开头。</segment></block>
  <block id="block-1"><segment>结尾。</segment></block>
</segmentation>
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)
    text = f"开头。{NO_SPLIT_START}这里。不要切！{NO_SPLIT_END}结尾。"

    result = LlmSentenceSegmenter(_config()).segment(text)

    assert [segment.text for segment in result.segments] == ["开头", "这里。不要切！", "结尾"]
    assert [segment.boundary for segment in result.segments] == ["llm", "protected", "llm"]


def test_llm_segmenter_falls_back_to_regex_xml_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
前面多余文本
<block id="block-0"><segment>第一段。</segment><segment>第二段。</segment></block>
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
<segmentation>
  <block id="block-0">
    <segment>满减后只要9.9元，</segment>
    <segment>还能叠加最高12元无门槛红包。</segment>
  </block>
</segmentation>
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
<segmentation>
  <block id="block-0">
    <segment>满减后只要 9.9 元，</segment>
    <segment>还能叠加最高12元无门槛红包。</segment>
  </block>
</segmentation>
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="不是原文顺序子串"):
        LlmSentenceSegmenter(_config()).segment("满减后只要9.9元，还能叠加最高12元无门槛红包。")


def test_llm_segmenter_raises_when_output_does_not_cover_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
<segmentation>
  <block id="block-0"><segment>第一段。</segment></block>
</segmentation>
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="未完整覆盖原文"):
        LlmSentenceSegmenter(_config()).segment("第一段。第二段。")


def test_llm_segmenter_writes_validation_diagnostics_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    diagnostics_path = tmp_path / "llm_segmentation_diagnostics.json"
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
<segmentation>
  <block id="block-0"><segment>如果你今天刚好要买纸巾。</segment></block>
</segmentation>
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="未直接拼接覆盖原文"):
        LlmSentenceSegmenter(_config(diagnostics_path=diagnostics_path)).segment(
            "长句测试：如果你今天刚好要买纸巾。"
        )

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["status"] == "failed"
    assert diagnostics["response"]["content"] == _FakeAsyncClient.response_content.strip()
    assert diagnostics["parsed"] == {"block-0": ["如果你今天刚好要买纸巾。"]}
    assert diagnostics["validation"]["failed_block_id"] == "block-0"
    assert diagnostics["validation"]["reason"] == "gap_before_segment"
    assert diagnostics["validation"]["gap"].startswith("长句测试")


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
