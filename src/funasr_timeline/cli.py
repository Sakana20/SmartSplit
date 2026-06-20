from __future__ import annotations

import argparse
from pathlib import Path

from funasr_timeline.asr.base import AsrService
from funasr_timeline.asr.mock_service import MockAsrService
from funasr_timeline.asr.paraformer_zh_service import (
    DEFAULT_PARAFORMER_MODEL_DIR,
    ParaformerZhAsrService,
)
from funasr_timeline.forced_alignment import (
    create_forced_alignment_service,
    load_aligner_config,
)
from funasr_timeline.forced_alignment.config import TimelineProvider
from funasr_timeline.logging import configure_logging, print_output_paths
from funasr_timeline.pipeline import run_pipeline, run_segmentation
from funasr_timeline.segmentation.factory import available_segmenters, create_segmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成以稿件为准的句子级音频时间轴。")
    parser.add_argument("--manuscript", required=True, type=Path, help=".txt 稿件路径")
    parser.add_argument("--audio", type=Path, help=".mp3 音频路径")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument(
        "--segment-only",
        action="store_true",
        help="只运行分句并输出可编辑分句文本，不执行 ASR、匹配和渲染。",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        help="使用人工编辑后的一行一句分句文件，替代自动分句结果继续执行后续流程。",
    )
    parser.add_argument(
        "--segmenter",
        choices=available_segmenters(),
        default="regex",
        help="分句实现。当前默认使用 regex。",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=Path("configs/llm-siliconflow.toml"),
        help="LLM 分句配置文件路径，仅在 --segmenter llm 时读取。",
    )
    parser.add_argument(
        "--timeline-provider",
        choices=["asr-fuzzy", "qwen3-forced", "hybrid"],
        help="时间轴来源。默认读取 aligner 配置；未配置时使用 hybrid。",
    )
    parser.add_argument(
        "--aligner-config",
        type=Path,
        default=Path("configs/aligner-qwen3.toml"),
        help="forced aligner 和 hybrid 时间轴配置文件路径。",
    )
    parser.add_argument(
        "--asr-provider",
        choices=["mock", "paraformer-zh"],
        help="ASR 服务实现；未提供时读取 aligner 配置。",
    )
    parser.add_argument(
        "--mock-word-timeline",
        type=Path,
        help="mock ASR 服务使用的 word_timeline.json 路径",
    )
    parser.add_argument(
        "--paraformer-model-dir",
        type=Path,
        default=DEFAULT_PARAFORMER_MODEL_DIR,
        help="本地 paraformer-zh 模型目录",
    )
    parser.add_argument(
        "--paraformer-device",
        default="mps",
        help="paraformer-zh 推理设备，例如 mps、cpu、cuda:0",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭调试日志，仅保留错误和 argparse 输出。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(quiet=args.quiet)
    segmenter = create_segmenter(args.segmenter, llm_config_path=args.llm_config)

    if args.segment_only:
        paths = run_segmentation(
            manuscript_path=args.manuscript,
            output_dir=args.output_dir,
            segmenter=segmenter,
        )
        print_output_paths(paths)
        return 0

    if args.audio is None:
        parser.error("完整流程需要提供 --audio；若只需要分句请使用 --segment-only")

    aligner_config = load_aligner_config(args.aligner_config)
    timeline_provider: TimelineProvider = args.timeline_provider or aligner_config.timeline.provider
    asr_provider = args.asr_provider or aligner_config.asr.provider

    asr_service: AsrService | None = None
    if timeline_provider in {"asr-fuzzy", "hybrid"}:
        if asr_provider == "mock":
            if args.mock_word_timeline is None:
                parser.error("--asr-provider mock 需要提供 --mock-word-timeline")
            asr_service = MockAsrService(args.mock_word_timeline)
        else:
            asr_service = ParaformerZhAsrService(
                model_dir=(
                    args.paraformer_model_dir
                    if args.paraformer_model_dir != DEFAULT_PARAFORMER_MODEL_DIR
                    else aligner_config.paraformer_zh.model_dir
                ),
                device=(
                    args.paraformer_device
                    if args.paraformer_device != "mps"
                    else aligner_config.paraformer_zh.device
                ),
            )

    forced_alignment_service = None
    if timeline_provider in {"qwen3-forced", "hybrid"}:
        forced_alignment_service = create_forced_alignment_service(aligner_config.qwen3_forced)

    paths = run_pipeline(
        manuscript_path=args.manuscript,
        audio_path=args.audio,
        output_dir=args.output_dir,
        asr_service=asr_service,
        segmenter=segmenter,
        segments_path=args.segments,
        timeline_provider=timeline_provider,
        forced_alignment_service=forced_alignment_service,
        forced_alignment_language=aligner_config.qwen3_forced.language,
        telemetry_config=aligner_config.telemetry,
    )
    print_output_paths(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
