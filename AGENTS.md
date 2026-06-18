# AGENTS.md

本项目用于构建一个基于统一 ASR 接口的音频时间轴处理流程：先从配音音频中生成字符级或 token 级时间轴，再将其与已有纯文本稿件进行顺序对齐，并通过顺序窗口 fuzzy 匹配合并为以原稿文本为准的句子级时间轴。当前已实现 mock ASR 服务和基于 FunASR `AutoModel` 的本地 `paraformer-zh` 服务。

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
- 句子切分保留可替换接口，当前内置 `regex` 和 `jieba-subtitle`。
- 稿件中可用 `[[NO_SPLIT]]...[[/NO_SPLIT]]` 标记保护不需要自动分句的片段。
- 数字归一化、领域词替换、同义词处理等复杂归一化暂不处理。
- ASR 能力通过统一接口接入，每个具体 ASR 服务实现该接口。
- 当前真实模型实现为 `paraformer-zh`，代码位于 `src/funasr_timeline/asr/paraformer_zh_service.py`。
- 本地 `paraformer-zh` 默认模型目录为 `/Users/sakana/PyEnv/paraformer`，默认推理设备为 macOS `mps`。
- 句子到 token 的匹配采用顺序窗口 fuzzy 匹配，适配 ground truth 文本生成音频的顺序稳定场景。
- 支持单独运行分句，输出 `editable_segments.txt`，并支持用编辑后的分句文件通过 `--segments` 继续完整流程。
- 最终句子时间范围必须保证相邻句子不重叠，并保留原始时间和修正诊断字段。
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

1. 通过统一 ASR 接口从音频生成字符级或 token 级时间轴。
2. 读取 `.txt` 稿件，并按可替换的句子切分接口生成句子片段。
3. 对稿件文本和 ASR 文本执行基础归一化。
4. 对稿件字符和 ASR 字符进行顺序全局对齐。
5. 将每个稿件句子按顺序窗口 fuzzy 匹配到 ASR token 时间轴。
6. 合并句子首尾 token 时间，并修正相邻句子的重叠时间范围。
7. 通过渲染接口导出 SRT 字幕。
8. 导出可复核的 JSON 结果、诊断报告和中间产物。

## 关键交付物

- `word_timeline.json`：ASR 生成的字符级或 token 级时间轴。
- `manuscript_segments.json`：稿件句子切分和归一化范围。
- `normalized_text.json`：稿件和 ASR 文本的归一化结果。
- `alignment.json`：全文顺序对齐结果和 offset 到 token 的映射。
- `sentence_timeline.json`：以稿件句子文本为准的句子级时间轴。
- `sentence_timeline.srt`：由句子级时间轴渲染得到的 SRT 字幕。
- `alignment_report.json`：低置信度匹配、未匹配字符、额外 ASR 内容和错配诊断。
- 可选的 `sentence_timeline.vtt`：后续阶段的字幕导出。
- 覆盖单元、集成和端到端行为的测试套件。
- lint、format、typecheck 和 test 的项目质量命令。

## 当前计划

当前确认后的实现计划见 `docs/implementation-plan.md`。
