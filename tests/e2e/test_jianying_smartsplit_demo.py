from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pytest

from funasr_timeline.segmentation import (
    NO_SPLIT_END,
    NO_SPLIT_START,
    remove_no_split_markers,
)
from funasr_timeline.segmentation.hanlp import PHRASE_BOUNDARIES
from funasr_timeline.segmentation.length import weighted_content_half_units

DEMO_TEXT = (
    "天气越来越热，出门走两步就容易出汗。无论是通勤赶地铁还是周末逛街，"
    "带上一个便携手持风扇都会舒服很多。"
    f"{NO_SPLIT_START}淘宝闪购正在发放最高12元无门槛红包{NO_SPLIT_END}，"
    "点击视频下方链接即可领取。\n\n"
    "很多学生放学路上最怕太阳暴晒，一个小巧的便携手持风扇放进书包就能随身带。"
    "课间、排队或者等公交的时候拿出来吹一吹，体验会好不少。"
    f"{NO_SPLIT_START}淘宝闪购红包活动进行中，最高12元无门槛红包别忘了领取"
    f"{NO_SPLIT_END}。\n\n"
    "如果你经常外出拍照、旅游或者看演出，手持风扇真的算是夏季实用小物。"
    "轻便不占地方，长时间使用也方便。"
    f"{NO_SPLIT_START}淘宝闪购限时补贴中{NO_SPLIT_END}，"
    "点击下方链接看看有没有适合你的款式。\n\n"
    "办公室空调开得不够足的时候，不少人都会准备一个随身风扇。"
    "桌面能放，出门能带，午休时也能派上用场。"
    f"{NO_SPLIT_START}现在淘宝闪购可领最高12元无门槛红包{NO_SPLIT_END}，"
    "点击视频下方链接直接了解。\n\n"
    "夏季出门最怕闷热难受，一个便携手持风扇随时都能带来凉爽体验。"
    "无论是在公园散步还是排队等餐，都能轻松使用。"
    f"{NO_SPLIT_START}淘宝闪购活动期间，最高12元无门槛红包正在发放"
    f"{NO_SPLIT_END}。\n\n"
    "打开视频下方链接，"
    f"{NO_SPLIT_START}先领取最高12元无门槛红包{NO_SPLIT_END}，"
    "再看看附近门店有哪些优惠。"
    "卷纸、雨伞、洗脸巾和水果都有不同力度补贴，"
    "部分商品支持半小时左右送达。\n\n"
    "最近需要购买iPhone15钢化膜、500ml饮料、2kg面粉、"
    "3.5L食用油或者99.9%除菌喷雾的话，可以顺便逛逛淘宝闪购。"
    "页面还有￥15.8优惠券、USD 5.99好物专区以及No.1热销商品推荐。\n\n"
    "夏季水果进入上市高峰期，荔枝、水蜜桃、杨梅和葡萄都很受欢迎。"
    "不少门店支持即时配送，下单后最快二十多分钟送到，"
    "对于临时采购来说会比较方便。\n\n"
    "活动期间还有各种优惠玩法，比如满29减8、满49减15、"
    "第二件半价以及3件8折等福利。搭配无门槛红包一起使用，"
    "买日用品或者零食都能省下一部分开销。\n\n"
    "记住几个常见入口：淘宝闪购、天猫超市、饿了么、视频下方链接、"
    "官方补贴、今日特价、限时秒杀和满减活动。下单前先领券，"
    "很多时候都能享受到更划算的价格。\n"
)

PROTECTED_DEMO_TEXTS = (
    "淘宝闪购正在发放最高12元无门槛红包",
    "淘宝闪购红包活动进行中，最高12元无门槛红包别忘了领取",
    "淘宝闪购限时补贴中",
    "现在淘宝闪购可领最高12元无门槛红包",
    "淘宝闪购活动期间，最高12元无门槛红包正在发放",
    "先领取最高12元无门槛红包",
)
DEMO_SPEECH_TEXT = remove_no_split_markers(DEMO_TEXT)

VOICE_ID = "BV005_streaming"
E2E_ENV_PATH = Path("configs/jianying-e2e.env")


@dataclass(frozen=True, slots=True)
class RealE2eSettings:
    draft_name: str
    voice_id: str
    jianying_scripts_path: Path | None
    aligner_config: Path
    llm_config: Path
    write_jianying_draft: bool


class CommandExecutionError(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(json.dumps(payload, ensure_ascii=False, indent=2))
        self.payload = payload


def test_run_command_streams_and_retains_subprocess_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('visible stdout', flush=True); "
                "print('visible stderr', file=sys.stderr, flush=True)"
            ),
        ]
    )

    captured = capsys.readouterr()
    assert "visible stdout" in captured.out
    assert "visible stderr" in captured.err
    assert result["stdout_tail"] == ["visible stdout"]
    assert result["stderr_tail"] == ["visible stderr"]


def test_demo_input_contains_protection_markers_before_real_e2e(tmp_path: Path) -> None:
    assert DEMO_TEXT.count(NO_SPLIT_START) == len(PROTECTED_DEMO_TEXTS)
    assert DEMO_TEXT.count(NO_SPLIT_END) == len(PROTECTED_DEMO_TEXTS)
    assert all(
        f"{NO_SPLIT_START}{text}{NO_SPLIT_END}" in DEMO_TEXT for text in PROTECTED_DEMO_TEXTS
    )
    assert NO_SPLIT_START not in DEMO_SPEECH_TEXT
    assert NO_SPLIT_END not in DEMO_SPEECH_TEXT
    assert all(text in DEMO_SPEECH_TEXT for text in PROTECTED_DEMO_TEXTS)

    paths = _prepare_real_workspace(tmp_path, tmp_path / "draft")
    manuscript_text = paths["manuscript"].read_text(encoding="utf-8")
    assert manuscript_text == DEMO_TEXT.strip() + "\n"


@pytest.mark.e2e_real
def test_jianying_demo_llm_qwen3_funasr_writes_detailed_diagnostics(
    tmp_path: Path,
) -> None:
    settings = _load_real_e2e_settings()
    draft_path = _reset_jianying_draft(settings)
    paths = _prepare_real_workspace(tmp_path, draft_path)

    audio_path, tts_meta = _generate_jianying_tts(paths["asset_dir"], settings)
    asr_audio_path, conversion_meta = _ensure_mp3_audio(audio_path, paths["timeline_dir"])

    command = [
        sys.executable,
        "-m",
        "funasr_timeline.cli",
        "--manuscript",
        str(paths["manuscript"]),
        "--audio",
        str(asr_audio_path),
        "--subtitle-alignment-audio",
        str(audio_path),
        "--output-dir",
        str(paths["timeline_dir"]),
        "--segmenter",
        "llm",
        "--llm-config",
        str(settings.llm_config),
        "--timeline-provider",
        "hybrid",
        "--aligner-config",
        str(settings.aligner_config),
    ]

    try:
        command_result = _run_command(command)
    except CommandExecutionError as error:
        _write_json(
            paths["timeline_dir"] / "e2e_failure_diagnostics.json",
            {
                "mode": "real-jianying-tts",
                "draft_name": settings.draft_name,
                "text_chars": len(DEMO_SPEECH_TEXT.strip()),
                "manuscript_text_chars": len(DEMO_TEXT.strip()),
                "protected_segment_count": len(PROTECTED_DEMO_TEXTS),
                "tts": tts_meta,
                "audio_conversion": conversion_meta,
                "command": error.payload,
            },
        )
        raise

    jianying_meta: dict[str, Any] = {"draft_path": str(draft_path), "srt_imported": False}
    if settings.write_jianying_draft:
        jianying_meta.update(_write_jianying_draft(draft_path, audio_path, paths["timeline_dir"]))

    diagnostics = _collect_timeline_diagnostics(
        paths["timeline_dir"],
        {
            "mode": "real-jianying-tts",
            "draft_name": settings.draft_name,
            "command": command_result,
            "text_chars": len(DEMO_SPEECH_TEXT.strip()),
            "manuscript_text_chars": len(DEMO_TEXT.strip()),
            "protected_segment_count": len(PROTECTED_DEMO_TEXTS),
            "tts": tts_meta,
            "audio_conversion": conversion_meta,
            "jianying": jianying_meta,
        },
    )
    _write_json(paths["timeline_dir"] / "e2e_diagnostics.json", diagnostics)
    _write_json(
        tmp_path / "summary.json",
        {
            "draft_path": str(draft_path),
            "audio": str(audio_path),
            "timeline_dir": str(paths["timeline_dir"]),
            "diagnostics": str(paths["timeline_dir"] / "e2e_diagnostics.json"),
        },
    )

    assert (paths["timeline_dir"] / "sentence_timeline.srt").exists()
    sentence_timeline = diagnostics["outputs"]["sentence_timeline"]
    telemetry = diagnostics["outputs"]["telemetry"]
    assert sentence_timeline["sentence_count"] > 0
    assert telemetry["timeline_provider"] == "hybrid"
    assert telemetry["forced_unit_count"] > 100
    assert telemetry["asr_token_count"] > 100
    assert telemetry["forced_normalized_text_match"] is True
    assert diagnostics["outputs"]["segmentation"]["protected_segment_count"] == len(
        PROTECTED_DEMO_TEXTS
    )
    assert diagnostics["outputs"]["segmentation"]["protected_texts"] == list(PROTECTED_DEMO_TEXTS)
    assert diagnostics["outputs"]["segmentation"]["overlong_nonprotected_segments"] == []
    assert diagnostics["outputs"]["segmentation"]["hanlp_phrase_boundary_violations"] == []

    timeline_texts = diagnostics["outputs"]["sentence_timeline"]["texts"]
    assert all(text in timeline_texts for text in PROTECTED_DEMO_TEXTS)
    llm_request_text = diagnostics["outputs"]["segmentation"]["llm_request_text"]
    assert all(text not in llm_request_text for text in PROTECTED_DEMO_TEXTS)


def _collect_timeline_diagnostics(timeline_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    sentence_timeline = _read_json(timeline_dir / "sentence_timeline.json")
    manuscript_segments = _read_json(timeline_dir / "manuscript_segments.json")
    llm_diagnostics = _read_json(timeline_dir / "llm_segmentation_diagnostics.json")
    telemetry = _read_json(timeline_dir / "telemetry.json")
    report = _read_json(timeline_dir / "alignment_report.json")
    srt_path = timeline_dir / "sentence_timeline.srt"

    status_counts: dict[str, int] = {}
    for item in sentence_timeline:
        status = str(item.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    start_deltas = [
        sentence.get("comparison", {}).get("start_delta_ms")
        for sentence in telemetry.get("sentences", [])
        if sentence.get("comparison", {}).get("start_delta_ms") is not None
    ]
    end_deltas = [
        sentence.get("comparison", {}).get("end_delta_ms")
        for sentence in telemetry.get("sentences", [])
        if sentence.get("comparison", {}).get("end_delta_ms") is not None
    ]
    protected_segments = [
        segment for segment in manuscript_segments if segment.get("boundary") == "protected"
    ]
    hard_max_content_length = int(
        llm_diagnostics.get("config", {}).get("segment_hard_max_content_length", 10)
    )
    overlong_nonprotected_segments = [
        {
            "index": segment.get("index"),
            "text": segment.get("text"),
            "segmenter": segment.get("segmenter"),
            "source_block_id": segment.get("source_block_id"),
            "content_length": weighted_content_half_units(str(segment.get("text", ""))) / 2,
        }
        for segment in manuscript_segments
        if segment.get("boundary") != "protected"
        and weighted_content_half_units(str(segment.get("text", ""))) > hard_max_content_length * 2
    ]
    hanlp_phrase_boundary_violations = [
        {
            "index": segment.get("index"),
            "text": segment.get("text"),
            "source_block_id": segment.get("source_block_id"),
        }
        for segment in manuscript_segments
        if segment.get("segmenter") == "hanlp"
        and set(str(segment.get("text", ""))) & PHRASE_BOUNDARIES
    ]
    llm_request_text = "\n".join(
        str(block.get("text", "")) for block in llm_diagnostics.get("request", {}).get("blocks", [])
    )

    return {
        "context": context,
        "outputs": {
            "sentence_timeline": {
                "path": str(timeline_dir / "sentence_timeline.json"),
                "sentence_count": len(sentence_timeline),
                "status_counts": status_counts,
                "first_text": sentence_timeline[0]["text"] if sentence_timeline else None,
                "last_text": sentence_timeline[-1]["text"] if sentence_timeline else None,
                "first_start_ms": sentence_timeline[0].get("start_ms")
                if sentence_timeline
                else None,
                "last_end_ms": sentence_timeline[-1].get("end_ms") if sentence_timeline else None,
                "texts": [item.get("text") for item in sentence_timeline],
            },
            "segmentation": {
                "manuscript_segments_path": str(timeline_dir / "manuscript_segments.json"),
                "llm_diagnostics_path": str(timeline_dir / "llm_segmentation_diagnostics.json"),
                "protected_segment_count": len(protected_segments),
                "protected_texts": [segment.get("text") for segment in protected_segments],
                "hard_max_content_length": hard_max_content_length,
                "overlong_nonprotected_segments": overlong_nonprotected_segments,
                "hanlp_phrase_boundary_violations": hanlp_phrase_boundary_violations,
                "llm_request_text": llm_request_text,
            },
            "srt": {
                "path": str(srt_path),
                "line_count": len(srt_path.read_text(encoding="utf-8").splitlines()),
            },
            "telemetry": {
                "path": str(timeline_dir / "telemetry.json"),
                "timeline_provider": telemetry.get("timeline_provider"),
                "forced_unit_count": telemetry.get("forced_alignment", {}).get("unit_count"),
                "forced_normalized_text_match": telemetry.get("forced_alignment", {}).get(
                    "normalized_text_match"
                ),
                "asr_token_count": telemetry.get("asr_fuzzy", {}).get("token_count"),
                "max_abs_start_delta_ms": _max_abs(start_deltas),
                "max_abs_end_delta_ms": _max_abs(end_deltas),
            },
            "report": {
                "path": str(timeline_dir / "alignment_report.json"),
                "low_confidence_count": len(report.get("low_confidence_sentences", [])),
                "unmatched_manuscript_count": len(
                    report.get("alignment", {}).get("unmatched_manuscript_indexes", [])
                ),
                "unmapped_asr_count": len(
                    report.get("alignment", {}).get("unmapped_asr_indexes", [])
                ),
            },
        },
    }


def _load_real_e2e_settings() -> RealE2eSettings:
    _load_env_file(E2E_ENV_PATH)
    jianying_scripts_path = os.environ.get("FUNASR_TIMELINE_E2E_JIANYING_SCRIPTS_PATH")
    return RealE2eSettings(
        draft_name=os.environ.get("FUNASR_TIMELINE_E2E_DRAFT", "SmartSplit_E2E_Test"),
        voice_id=os.environ.get("FUNASR_TIMELINE_E2E_VOICE_ID", VOICE_ID),
        jianying_scripts_path=Path(jianying_scripts_path) if jianying_scripts_path else None,
        aligner_config=Path(
            os.environ.get("FUNASR_TIMELINE_E2E_ALIGNER_CONFIG", "configs/aligner-qwen3.toml")
        ),
        llm_config=Path(
            os.environ.get("FUNASR_TIMELINE_E2E_LLM_CONFIG", "configs/llm-siliconflow.toml")
        ),
        write_jianying_draft=os.environ.get("FUNASR_TIMELINE_E2E_WRITE_DRAFT", "1") != "0",
    )


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _reset_jianying_draft(settings: RealE2eSettings) -> Path:
    _add_optional_jianying_scripts_to_path(settings.jianying_scripts_path)
    from jy_wrapper import JyProject, get_default_drafts_root

    drafts_root = Path(get_default_drafts_root()).resolve()
    draft_path = (drafts_root / settings.draft_name).resolve()
    if draft_path.exists():
        if drafts_root not in draft_path.parents:
            raise RuntimeError(f"refusing to delete draft outside Jianying root: {draft_path}")
        shutil.rmtree(draft_path)
    result = JyProject(settings.draft_name, width=1080, height=1920, overwrite=True).save()
    draft_path = Path(result["draft_path"])
    (draft_path / "smartsplit_assets").mkdir(parents=True, exist_ok=True)
    return draft_path


def _prepare_real_workspace(tmp_path: Path, draft_path: Path) -> dict[str, Path]:
    workspace = tmp_path / "real_jianying_e2e"
    input_dir = workspace / "inputs"
    timeline_dir = workspace / "timeline"
    asset_dir = draft_path / "smartsplit_assets"
    input_dir.mkdir(parents=True)
    timeline_dir.mkdir(parents=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    manuscript_path = input_dir / "gt.txt"
    manuscript_path.write_text(DEMO_TEXT.strip() + "\n", encoding="utf-8")
    return {
        "workspace": workspace,
        "manuscript": manuscript_path,
        "timeline_dir": timeline_dir,
        "asset_dir": asset_dir,
    }


def _generate_jianying_tts(
    asset_dir: Path,
    settings: RealE2eSettings,
) -> tuple[Path, dict[str, Any]]:
    _add_optional_jianying_scripts_to_path(settings.jianying_scripts_path)
    from universal_tts import generate_voice_with_meta

    audio_path = asset_dir / "voice_smartsplit_e2e.ogg"

    async def generate() -> tuple[str | None, str | None]:
        return await generate_voice_with_meta(
            DEMO_SPEECH_TEXT.strip(),
            str(audio_path),
            settings.voice_id,
            allow_fallback=False,
            sami_retries=2,
        )

    actual_path, backend = asyncio.run(generate())
    if actual_path is None:
        raise RuntimeError("Jianying TTS returned no audio path")
    actual_audio_path = Path(actual_path)
    if actual_audio_path != audio_path:
        shutil.copy2(actual_audio_path, audio_path)
    return audio_path, {
        "voice_id": settings.voice_id,
        "backend": backend,
        "audio_path": str(audio_path),
        "actual_audio_path": str(actual_audio_path),
    }


def _ensure_mp3_audio(audio_path: Path, timeline_dir: Path) -> tuple[Path, dict[str, Any]]:
    if audio_path.suffix.lower() == ".mp3":
        return audio_path, {"converted": False, "audio_path": str(audio_path)}

    converted_path = timeline_dir / f"{audio_path.stem}.mp3"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(converted_path),
    ]
    result = _run_command(command)
    return converted_path, {
        "converted": True,
        "source_audio_path": str(audio_path),
        "audio_path": str(converted_path),
        "command": result,
    }


def _write_jianying_draft(
    draft_path: Path,
    audio_path: Path,
    timeline_dir: Path,
) -> dict[str, Any]:
    _add_optional_jianying_scripts_to_path(None)
    from jy_wrapper import JyProject

    material_audio_path = draft_path / "materials" / "audio" / audio_path.name
    material_audio_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_path, material_audio_path)

    project = JyProject(draft_path.name, width=1080, height=1920, overwrite=False)
    audio_segment = project.add_audio_safe(
        str(audio_path),
        start_time="0s",
        track_name="SmartSplit_E2E_Audio",
    )
    if audio_segment is None:
        raise RuntimeError(f"failed to import audio into Jianying draft: {audio_path}")
    project.script.import_srt(
        str(timeline_dir / "sentence_timeline.srt"),
        track_name="SmartSplit_E2E_Subtitles",
    )
    result = project.save()
    return {
        "draft_path": str(result["draft_path"]),
        "material_audio_path": str(material_audio_path),
        "srt_path": str(timeline_dir / "sentence_timeline.srt"),
        "srt_imported": True,
    }


def _add_optional_jianying_scripts_to_path(scripts_path: Path | None) -> None:
    if scripts_path is None:
        return
    if not scripts_path.exists():
        raise FileNotFoundError(f"Jianying scripts path not found: {scripts_path}")
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))


def _run_command(command: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stdout_thread = threading.Thread(
        target=_tee_stream,
        args=(process.stdout, sys.stdout, stdout_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_tee_stream,
        args=(process.stderr, sys.stderr, stderr_lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    payload = {
        "command": _safe_command(command),
        "returncode": returncode,
        "stdout_tail": stdout_lines,
        "stderr_tail": stderr_lines,
    }
    if returncode != 0:
        raise CommandExecutionError(payload)
    return payload


def _tee_stream(source: TextIO, destination: TextIO, lines: list[str]) -> None:
    try:
        for line in source:
            lines.append(line.rstrip("\r\n"))
            destination.write(line)
            destination.flush()
    finally:
        source.close()


def _safe_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        redacted.append(item)
        if item in {"--llm-api-key"}:
            redact_next = True
    return redacted


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _max_abs(values: list[int]) -> int | None:
    return max((abs(value) for value in values), default=None)
