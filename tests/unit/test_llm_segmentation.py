from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from funasr_timeline.segmentation import NO_SPLIT_END, NO_SPLIT_START
from funasr_timeline.segmentation.llm import (
    LlmSegmentationConfig,
    LlmSentenceSegmenter,
    load_llm_segmentation_config,
)
from funasr_timeline.segmentation.regex import RegexSentenceSegmenter


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
    active_requests = 0
    max_active_requests = 0

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
        type(self).active_requests += 1
        type(self).max_active_requests = max(
            type(self).max_active_requests,
            type(self).active_requests,
        )
        await asyncio.sleep(0)
        type(self).active_requests -= 1
        return _FakeResponse(content, self.response_payload)


def test_llm_segmenter_splits_multiple_blocks_in_parallel_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.active_requests = 0
    _FakeAsyncClient.max_active_requests = 0
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = ["第一段。\n第二段。", "第三段，\n继续。"]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment("第一段。第二段。\n第三段，继续。")

    assert [segment.text for segment in result.segments] == ["第一段", "第二段", "第三段", "继续"]
    assert [segment.paragraph_index for segment in result.segments] == [0, 0, 1, 1]
    assert len(_FakeAsyncClient.requests) == 2
    assert _FakeAsyncClient.max_active_requests == 2
    assert _FakeAsyncClient.requests[0]["url"] == "https://example.test/v1/chat/completions"
    payload = _FakeAsyncClient.requests[0]["json"]
    assert payload["model"] == "test-model"
    assert payload["enable_thinking"] is False
    prompt = payload["messages"][0]["content"]
    assert "第一段。第二段。" in prompt
    assert "第三段，继续。" not in prompt
    second_prompt = _FakeAsyncClient.requests[1]["json"]["messages"][0]["content"]
    assert "第三段，继续。" in second_prompt
    assert "不要输出 XML" in prompt
    assert "不要输出 JSON" in prompt
    assert "只允许插入分句边界" in prompt
    assert "忽略标点与空格完成内容校验" in prompt
    assert "去掉标点、空格和分隔符后拼接" in prompt
    assert "折算长度优先控制在 4 到 8 个汉字左右" in prompt
    assert "折算长度必须不超过 10 个汉字" in prompt
    assert "英文和数字每两个计 1" in prompt
    assert "特别" in prompt
    assert "再比如" in prompt
    assert "按原稿定位回填" in prompt


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


def test_llm_segmenter_retries_only_failed_blocks_and_merges_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    diagnostics_path = tmp_path / "llm_segmentation_diagnostics.json"
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = [
        "第一段。",
        "天气热的时候最怕空气闷闷的，这款便携手持风扇轻松带来凉爽体验。",
        "天气热的时候\n最怕空气闷闷的，\n这款便携手持风扇\n轻松带来凉爽体验。",
    ]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config(diagnostics_path=diagnostics_path)).segment(
        "第一段。\n天气热的时候最怕空气闷闷的，这款便携手持风扇轻松带来凉爽体验。"
    )

    assert [segment.text for segment in result.segments] == [
        "第一段",
        "天气热的时候",
        "最怕空气闷闷的",
        "这款便携手持风扇",
        "轻松带来凉爽体验",
    ]
    assert len(_FakeAsyncClient.requests) == 3
    retry_prompt = _FakeAsyncClient.requests[2]["json"]["messages"][0]["content"]
    assert '<input_block id="block-0">' not in retry_prompt
    assert "第一段。" not in retry_prompt
    assert '<input_block id="block-1">' in retry_prompt
    assert "天气热的时候最怕空气闷闷的" in retry_prompt

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["parsed"]["block-0"] == ["第一段。"]
    assert diagnostics["attempts"][0]["requested_block_ids"] == ["block-0"]
    assert diagnostics["attempts"][0]["accepted_block_ids"] == ["block-0"]
    assert diagnostics["attempts"][1]["requested_block_ids"] == ["block-1"]
    assert diagnostics["attempts"][1]["accepted_block_ids"] == []
    assert diagnostics["attempts"][2]["requested_block_ids"] == ["block-1"]
    assert diagnostics["attempts"][2]["accepted_block_ids"] == ["block-1"]


def test_llm_segmenter_reports_frozen_and_pending_blocks_after_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    diagnostics_path = tmp_path / "llm_segmentation_diagnostics.json"
    long_block = "天气热的时候最怕空气闷闷的，这款便携手持风扇轻松带来凉爽体验。"
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = [
        "第一段。",
        long_block,
        long_block,
    ]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="重试后仍未通过校验"):
        LlmSentenceSegmenter(
            _config(diagnostics_path=diagnostics_path, max_retries=1),
            fallback_segmenter=RegexSentenceSegmenter(),
            raise_on_error=True,
        ).segment(f"第一段。\n{long_block}")

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    validation = diagnostics["validation"]
    assert validation["accepted"] == {"block-0": ["第一段。"]}
    assert validation["pending_block_ids"] == ["block-1"]
    assert validation["attempts"][0]["requested_block_ids"] == ["block-0"]
    assert validation["attempts"][1]["requested_block_ids"] == ["block-1"]
    assert validation["attempts"][2]["requested_block_ids"] == ["block-1"]
    assert validation["last_error"]["original_blocks"] == [{"id": "block-1", "text": long_block}]
    assert "第一段。" not in _FakeAsyncClient.requests[2]["json"]["messages"][0]["content"]


def test_llm_segmenter_falls_back_only_failed_block_and_records_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    diagnostics_path = tmp_path / "llm_segmentation_diagnostics.json"
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = ["第一段。", "错误内容。"]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(
        _config(diagnostics_path=diagnostics_path, max_retries=0),
        fallback_segmenter=RegexSentenceSegmenter(),
    ).segment("第一段。\n第二段。第三段。")

    assert [segment.text for segment in result.segments] == ["第一段", "第二段。", "第三段。"]
    assert [segment.segmenter for segment in result.segments] == ["llm", "regex", "regex"]
    assert [segment.source_block_id for segment in result.segments] == [
        "block-0",
        "block-1",
        "block-1",
    ]
    assert [segment.index for segment in result.segments] == [0, 1, 2]
    assert all(
        result.text[segment.char_start : segment.char_end] == segment.text
        for segment in result.segments
    )

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["status"] == "validated_with_fallback"
    assert diagnostics["validation"]["block_strategies"] == {
        "block-0": "llm",
        "block-1": "regex",
    }
    assert diagnostics["failures"]["block-1"]["reason"] == "retry_exhausted"


def test_llm_segmenter_retries_request_errors_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    class _FlakyAsyncClient(_FakeAsyncClient):
        call_count = 0

        async def post(self, url: str, **kwargs: object) -> _FakeResponse:
            type(self).call_count += 1
            if type(self).call_count == 1:
                raise httpx.ConnectError("temporary connection failure")
            return await super().post(url, **kwargs)

    _FlakyAsyncClient.call_count = 0
    _FlakyAsyncClient.requests = []
    _FlakyAsyncClient.response_payload = None
    _FlakyAsyncClient.response_contents = []
    _FlakyAsyncClient.response_content = "第一段。"
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FlakyAsyncClient)

    result = LlmSentenceSegmenter(
        _config(max_retries=1),
        fallback_segmenter=RegexSentenceSegmenter(),
    ).segment("第一段。")

    assert [segment.text for segment in result.segments] == ["第一段"]
    assert result.segments[0].segmenter == "llm"
    assert _FlakyAsyncClient.call_count == 2


def test_llm_segmenter_length_check_counts_two_english_digits_as_one_han_char(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = []
    _FakeAsyncClient.response_contents = [
        "iPhone15手机壳、500ml矿泉水、2kg大米。",
        "iPhone15手机壳、\n500ml矿泉水、\n2kg大米。",
    ]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment("iPhone15手机壳、500ml矿泉水、2kg大米。")

    assert [segment.text for segment in result.segments] == [
        "iPhone15手机壳",
        "500ml矿泉水",
        "2kg大米",
    ]
    assert len(_FakeAsyncClient.requests) == 2
    retry_prompt = _FakeAsyncClient.requests[1]["json"]["messages"][0]["content"]
    assert "content_length" in retry_prompt
    assert "英文和数字每两个计 1" in retry_prompt


def test_llm_segmenter_keeps_protected_block_as_single_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = ["开头", "结尾。"]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)
    text = f"开头。{NO_SPLIT_START}这里。不要切！{NO_SPLIT_END}结尾。"

    result = LlmSentenceSegmenter(_config()).segment(text)

    assert [segment.text for segment in result.segments] == ["开头", "这里。不要切！", "结尾"]
    assert [segment.boundary for segment in result.segments] == ["llm", "protected", "llm"]


def test_llm_segmenter_forces_boundaries_around_protected_blocks_with_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = [
        "很多人买雨伞\n都会关注便携和耐用，\n这种日常用品\n用得顺手才重要。\n放包里不占空间，\n出门更省心",
        "点击下方链接了解更多。",
        "周末逛街、旅游\n或者上下班通勤，\n随身带把雨伞\n总能多一份保障。\n遇到天气变化\n也不用临时找地方躲雨",
        "点击视频下方链接选购。",
        "出门最怕下雨没带伞，\n不仅影响行程\n还容易淋湿。\n准备一把轻便雨伞，\n日常使用更方便",
        "点击视频下方链接看看。",
    ]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)
    text = (
        "很多人买雨伞都会关注便携和耐用，这种日常用品用得顺手才重要。"
        "放包里不占空间，出门更省心。"
        f"{NO_SPLIT_START}淘宝闪购最高12元无门槛红包可领取{NO_SPLIT_END}"
        "，点击下方链接了解更多。\n\n"
        "周末逛街、旅游或者上下班通勤，随身带把雨伞总能多一份保障。"
        "遇到天气变化也不用临时找地方躲雨。"
        f"{NO_SPLIT_START}淘宝闪购可领最高12元无门槛红包{NO_SPLIT_END}"
        "，点击视频下方链接选购。\n\n"
        "出门最怕下雨没带伞，不仅影响行程还容易淋湿。"
        "准备一把轻便雨伞，日常使用更方便。"
        f"{NO_SPLIT_START}淘宝闪购最高12元无门槛红包正在领取中{NO_SPLIT_END}"
        "，点击视频下方链接看看。"
    )

    result = LlmSentenceSegmenter(_config()).segment(text)

    assert [segment.text for segment in result.segments] == [
        "很多人买雨伞",
        "都会关注便携和耐用",
        "这种日常用品",
        "用得顺手才重要",
        "放包里不占空间",
        "出门更省心",
        "淘宝闪购最高12元无门槛红包可领取",
        "点击下方链接了解更多",
        "周末逛街、旅游",
        "或者上下班通勤",
        "随身带把雨伞",
        "总能多一份保障",
        "遇到天气变化",
        "也不用临时找地方躲雨",
        "淘宝闪购可领最高12元无门槛红包",
        "点击视频下方链接选购",
        "出门最怕下雨没带伞",
        "不仅影响行程",
        "还容易淋湿",
        "准备一把轻便雨伞",
        "日常使用更方便",
        "淘宝闪购最高12元无门槛红包正在领取中",
        "点击视频下方链接看看",
    ]
    assert [segment.boundary for segment in result.segments].count("protected") == 3
    assert all(
        result.text[segment.char_start : segment.char_end] == segment.text
        for segment in result.segments
    )
    assert all(
        previous.char_end <= current.char_start
        for previous, current in zip(result.segments, result.segments[1:], strict=False)
    )

    prompts = [request["json"]["messages"][0]["content"] for request in _FakeAsyncClient.requests]
    combined_prompts = "\n".join(prompts)
    assert NO_SPLIT_START not in combined_prompts
    assert NO_SPLIT_END not in combined_prompts
    assert "淘宝闪购最高12元无门槛红包可领取" not in combined_prompts
    assert "，点击下方链接了解更多。" not in combined_prompts
    assert "点击下方链接了解更多。" in combined_prompts


def test_llm_segmenter_drops_punctuation_only_seam_between_protected_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_contents = ["前文", "后文。"]
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)
    text = (
        f"前文。{NO_SPLIT_START}保护一{NO_SPLIT_END}，；"
        f"{NO_SPLIT_START}保护二{NO_SPLIT_END}。后文。"
    )

    result = LlmSentenceSegmenter(_config()).segment(text)

    assert [segment.text for segment in result.segments] == ["前文", "保护一", "保护二", "后文"]
    assert [segment.boundary for segment in result.segments] == [
        "llm",
        "protected",
        "protected",
        "llm",
    ]
    prompt = _FakeAsyncClient.requests[0]["json"]["messages"][0]["content"]
    assert "，；" not in prompt


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
还能叠加
最高12元无门槛红包。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment(
        "满减后只要9.9元，还能叠加最高12元无门槛红包。"
    )

    assert [segment.text for segment in result.segments] == [
        "满减后只要9.9元",
        "还能叠加",
        "最高12元无门槛红包",
    ]


def test_llm_segmenter_accepts_added_spaces_and_maps_back_to_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = None
    _FakeAsyncClient.response_content = """
满减后只要 9.9 元，
还能叠加
最高12元无门槛红包。
"""
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    result = LlmSentenceSegmenter(_config()).segment(
        "满减后只要9.9元，还能叠加最高12元无门槛红包。"
    )

    assert [segment.text for segment in result.segments] == [
        "满减后只要9.9元",
        "还能叠加",
        "最高12元无门槛红包",
    ]
    assert all(" " not in segment.text for segment in result.segments)


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
        LlmSentenceSegmenter(
            _config(),
            fallback_segmenter=RegexSentenceSegmenter(),
            raise_on_error=True,
        ).segment("第一段。第二段。")


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
        LlmSentenceSegmenter(
            _config(diagnostics_path=diagnostics_path),
            fallback_segmenter=RegexSentenceSegmenter(),
            raise_on_error=True,
        ).segment("长句测试：如果你今天刚好要买纸巾。")

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["status"] == "failed"
    assert diagnostics["parsed"] == {}
    assert diagnostics["validation"]["reason"] == "retry_exhausted"
    last_error = diagnostics["validation"]["last_error"]
    assert last_error["diagnostics"]["failed_block_id"] == "block-0"
    assert last_error["diagnostics"]["reason"] == "normalized_content_not_equal"
    assert last_error["previous_output"] == _FakeAsyncClient.response_content.strip()


def test_llm_segmenter_reports_empty_content_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import funasr_timeline.segmentation.llm as llm_segmentation

    _FakeAsyncClient.response_payload = {
        "choices": [{"finish_reason": "length", "message": {"content": ""}}]
    }
    monkeypatch.setattr(llm_segmentation.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ValueError, match="重试后仍未通过校验"):
        LlmSentenceSegmenter(
            _config(),
            fallback_segmenter=RegexSentenceSegmenter(),
            raise_on_error=True,
        ).segment("第一段。")


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


def _config(
    diagnostics_path: Path | None = None,
    max_retries: int = 3,
) -> LlmSegmentationConfig:
    return LlmSegmentationConfig(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="TEST_LLM_KEY",
        api_key="test-key",
        diagnostics_path=diagnostics_path,
        max_retries=max_retries,
    )
