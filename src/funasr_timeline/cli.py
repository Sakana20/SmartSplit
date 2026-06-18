from __future__ import annotations

import argparse
from pathlib import Path

from funasr_timeline.asr.base import AsrService
from funasr_timeline.asr.mock_service import MockAsrService
from funasr_timeline.asr.paraformer_zh_service import (
    DEFAULT_PARAFORMER_MODEL_DIR,
    ParaformerZhAsrService,
)
from funasr_timeline.pipeline import run_pipeline, run_segmentation
from funasr_timeline.segmentation import available_segmenters, create_segmenter


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
        "--asr-provider",
        choices=["mock", "paraformer-zh"],
        default="mock",
        help="ASR 服务实现。第一阶段默认使用 mock。",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    segmenter = create_segmenter(args.segmenter)

    if args.segment_only:
        paths = run_segmentation(
            manuscript_path=args.manuscript,
            output_dir=args.output_dir,
            segmenter=segmenter,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0

    if args.audio is None:
        parser.error("完整流程需要提供 --audio；若只需要分句请使用 --segment-only")

    if args.asr_provider == "mock":
        if args.mock_word_timeline is None:
            parser.error("--asr-provider mock 需要提供 --mock-word-timeline")
        asr_service: AsrService = MockAsrService(args.mock_word_timeline)
    else:
        asr_service = ParaformerZhAsrService(
            model_dir=args.paraformer_model_dir,
            device=args.paraformer_device,
        )

    paths = run_pipeline(
        manuscript_path=args.manuscript,
        audio_path=args.audio,
        output_dir=args.output_dir,
        asr_service=asr_service,
        segmenter=segmenter,
        segments_path=args.segments,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
