from __future__ import annotations

import os
import re
import sys
from collections.abc import MutableMapping
from pathlib import Path

from funasr_timeline.asr.paraformer_zh_service import DEFAULT_PARAFORMER_MODEL_DIR
from funasr_timeline.cli import CliDefaults, run_cli
from funasr_timeline.render.srt import (
    DEFAULT_SUBTITLE_GAP_THRESHOLD_MS,
    DEFAULT_SUBTITLE_MIN_DURATION_MS,
)

DEFAULT_E2E_ENV_PATH = Path("configs/jianying-e2e.env")
DEFAULT_LLM_CONFIG_PATH = Path("configs/llm-siliconflow.toml")
DEFAULT_ALIGNER_CONFIG_PATH = Path("configs/aligner-qwen3.toml")
_ENV_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def load_env_file(path: Path, environ: MutableMapping[str, str] | None = None) -> None:
    """Load plain KEY=VALUE entries without executing the file as shell code."""
    target = os.environ if environ is None else environ
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"环境文件第 {line_number} 行不是 KEY=VALUE：{path}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"环境文件第 {line_number} 行包含无效变量名 {key!r}：{path}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        target.setdefault(key, value)


def build_smartsplit_defaults(environ: MutableMapping[str, str] | None = None) -> CliDefaults:
    source = os.environ if environ is None else environ
    return CliDefaults(
        segmenter="llm",
        llm_config=Path(
            source.get("SMARTSPLIT_LLM_CONFIG")
            or source.get("FUNASR_TIMELINE_E2E_LLM_CONFIG")
            or DEFAULT_LLM_CONFIG_PATH
        ),
        llm_fallback_segmenter="hanlp",
        timeline_provider="hybrid",
        aligner_config=Path(
            source.get("SMARTSPLIT_ALIGNER_CONFIG")
            or source.get("FUNASR_TIMELINE_E2E_ALIGNER_CONFIG")
            or DEFAULT_ALIGNER_CONFIG_PATH
        ),
        asr_provider="paraformer-zh",
        paraformer_model_dir=Path(
            source.get("SMARTSPLIT_PARAFORMER_MODEL_DIR") or DEFAULT_PARAFORMER_MODEL_DIR
        ),
        paraformer_device=source.get("SMARTSPLIT_PARAFORMER_DEVICE") or "mps",
        log_level="INFO",
        subtitle_gap_threshold_ms=DEFAULT_SUBTITLE_GAP_THRESHOLD_MS,
        subtitle_min_duration_ms=DEFAULT_SUBTITLE_MIN_DURATION_MS,
    )


def main(argv: list[str] | None = None) -> int:
    explicit_env_path = os.environ.get("SMARTSPLIT_E2E_ENV")
    env_path = Path(explicit_env_path) if explicit_env_path else DEFAULT_E2E_ENV_PATH
    if env_path.is_file():
        try:
            load_env_file(env_path)
        except (OSError, ValueError) as error:
            print(f"SmartSplit 环境文件读取失败：{error}", file=sys.stderr)
            return 2
    elif explicit_env_path:
        print(f"SmartSplit 环境文件不存在：{env_path}", file=sys.stderr)
        return 2

    return run_cli(
        argv,
        defaults=build_smartsplit_defaults(),
        prog="smartsplit",
        description="从现成稿件和音频生成 SmartSplit SRT 字幕及可复核诊断。",
    )


if __name__ == "__main__":
    raise SystemExit(main())
