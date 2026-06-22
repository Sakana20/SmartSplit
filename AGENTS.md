# AGENTS.md

本项目用于构建一个以稿件为准、可复核的音频时间轴处理流程：既可从统一 ASR 接口生成 token 时间轴并执行顺序 fuzzy 匹配，也可使用 Qwen3 ForcedAligner 将稿件直接对齐到 TTS 音频。当前实现支持 `asr-fuzzy`、`qwen3-forced` 和以 forced alignment 为主时间来源的 `hybrid` 三种策略。

## 工作原则

- 未经用户确认实现计划和开放问题前，不开始编写实现代码。
- 最终句子文本默认以稿件原文为唯一来源，除非用户明确要求改用 ASR 文本。
- 优先采用确定性、可检查的对齐逻辑，避免不可解释的黑盒启发式处理。
- 保留中间产物，方便人工复核错配、漏配、额外识别内容和低置信度片段。
- 变更应保持聚焦，并在 `docs/` 中记录关键假设、阶段边界和行为变化。
- 仓库行为、实现细节和文档必须同步演进。

## 当前阶段边界

当前阶段目标是跑通端到端的可复核流程，并为后续音频格式、字幕导出和更复杂匹配策略保留接口：

- 稿件输入使用 `.txt` 纯文本文件，内容为分段文本。
- 音频输入第一阶段先跑通 `.mp3`。
- 多种常见音频格式如 `.wav`、`.ogg` 等，以及统一音频格式转换能力先保留为待做事项。
- 句子切分保留可替换接口，当前内置 `regex`、`hanlp`、`jieba-subtitle` 和 `llm`。
- LLM 分句按 block 并发请求和独立重试；失败 block 默认使用 `hanlp` fallback，成功 block 不重跑。
- 稿件中可用 `[[NO_SPLIT]]...[[/NO_SPLIT]]` 标记保护不需要自动分句的片段。
- 数字归一化、领域词替换、同义词处理等复杂归一化暂不处理。
- ASR 能力通过统一接口接入，每个具体 ASR 服务实现该接口。
- 当前真实模型实现为 `paraformer-zh`，代码位于 `src/funasr_timeline/asr/paraformer_zh_service.py`。
- 本地 `paraformer-zh` 默认模型目录为 `/Users/sakana/PyEnv/paraformer`，默认推理设备为 macOS `mps`。
- 当前 forced alignment 实现为本地 `Qwen3-ForcedAligner-0.6B`，默认模型目录为 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`，默认使用 `mps` 和 `bfloat16`。
- 句子到 token 的匹配采用顺序窗口 fuzzy 匹配，适配 ground truth 文本生成音频的顺序稳定场景。
- 支持单独运行分句，输出 `editable_segments.txt`，并支持用编辑后的分句文件通过 `--segments` 继续完整流程。
- 最终句子时间范围必须保证相邻句子不重叠，并保留原始时间和修正诊断字段。
- SRT 默认将首条有效字幕对齐音频起点、末条有效字幕对齐音频结尾，填充不超过 20 帧（精确计算 667ms，约 670ms）的短间隙，并尽量把不足 200ms 的字幕延长到最短时长；渲染修正只作用于 SRT 副本并写入 `subtitle_render_report.json`。
- 输出应尽可能丰富，包含对齐、诊断和中间字段，供后续分析与调整。

## Python 项目规范

- 使用 `uv` 进行依赖管理和命令执行。
- 优先采用 `pyproject.toml` 驱动的项目结构，并显式区分运行时依赖和开发依赖。
- 添加并维护 lint、format、typecheck 和 test 命令。
- lint、typecheck 和 test 失败默认视为阻塞问题，除非用户明确接受风险。
- 遵循 Python 最佳实践：公共接口带类型标注，模块职责小而清晰，错误处理明确，输出确定，不引入隐藏的网络或文件系统副作用。

## 测试规范

- 实现代码必须配套测试。
- 纯逻辑使用单元测试覆盖。
- 稿件到 ASR 时间轴的对齐流程使用集成测试覆盖。
- 命令行或端到端流程使用小型 fixture 文件覆盖。
- 常规测试保持确定性，不依赖大型模型下载；FunASR 或其他真实 ASR 输出应优先使用 mock 或 fixture。

## 预期流程

1. 根据时间轴策略，通过统一 ASR 接口生成 token 时间轴，和/或通过 forced alignment 接口将稿件对齐到音频。
2. 读取 `.txt` 稿件，并按可替换的句子切分接口生成句子片段。
3. 对稿件文本和 ASR 文本执行基础归一化。
4. 对稿件字符和 ASR 字符进行顺序全局对齐。
5. 在 ASR fuzzy 分支中，将每个稿件句子按顺序窗口 fuzzy 匹配到 ASR token 时间轴；在 forced 分支中按归一化 offset 映射句子时间。
6. 根据 `asr-fuzzy`、`qwen3-forced` 或 `hybrid` 策略选择最终句子时间，并修正相邻句子的重叠范围。
7. 通过渲染接口导出 SRT 字幕。
8. 导出可复核的 JSON 结果、诊断报告和中间产物。

## 关键交付物

- `word_timeline.json`：ASR 生成的字符级或 token 级时间轴。
- `manuscript_segments.json`：稿件句子切分和归一化范围。
- `normalized_text.json`：稿件和 ASR 文本的归一化结果。
- `alignment.json`：全文顺序对齐结果和 offset 到 token 的映射。
- `sentence_timeline.json`：以稿件句子文本为准的句子级时间轴。
- `sentence_timeline.srt`：由句子级时间轴渲染得到的 SRT 字幕。
- `subtitle_render_report.json`：SRT 首尾对齐、短间隙填充、短字幕延长和未完全修复项的诊断。
- `alignment_report.json`：低置信度匹配、未匹配字符、额外 ASR 内容和错配诊断。
- `forced_alignment.json`：forced aligner 标准化单元和文本一致性诊断，仅 forced/hybrid 模式输出。
- `telemetry.json`：forced 与 ASR fuzzy 分支对比，仅 forced/hybrid 模式输出。
- 可选的 `sentence_timeline.vtt`：后续阶段的字幕导出。
- 覆盖单元、集成和端到端行为的测试套件。
- lint、format、typecheck 和 test 的项目质量命令。

## 当前计划

当前确认后的实现计划见 `docs/implementation-plan.md`。
