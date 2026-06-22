# 实现计划

## 项目目标

本项目要实现一个可检查、可扩展的音频时间轴生成流程：

1. 从配音音频中生成字符级或 token 级 ASR 时间轴。
2. 将 ASR 文本与已有稿件纯文本进行顺序对齐。
3. 可选使用 Qwen3 forced aligner 将真实稿件文本直接对齐到 TTS 音频。
4. 以稿件原文为准，将 ASR fuzzy 或 forced alignment 时间范围合并为句子级时间范围。
5. 输出丰富的 JSON 中间产物和诊断信息，供后续调参、复核和扩展。

当前重点是跑通 `.txt` 稿件和 `.mp3` 音频的可复核主流程，并为后续多音频格式、音频转换、更多 ASR/forced alignment 服务和字幕导出留下清晰接口。当前真实模型实现包括本地 `paraformer-zh` ASR 和本地 `Qwen3-ForcedAligner-0.6B`。

## 项目工具

项目使用 `uv` 进行依赖管理和命令执行。

推荐基础结构：

- 使用 `pyproject.toml` 作为依赖、包信息和工具配置的单一来源。
- 运行时依赖覆盖 ASR 接口、文本归一化、对齐、JSON 导出等能力。
- 开发依赖覆盖测试、lint、format 和 typecheck。
- 依赖确定后提交锁文件。

推荐命令形态：

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

实际命令可以在包结构建立后微调，但必须保持文档和项目配置同步。

## 当前实现范围

当前已实现最小可用但信息丰富的离线流程。

### 实现状态

当前已实现：

- `pyproject.toml` 驱动的 `uv` 项目结构。
- `funasr-timeline` 命令行入口。
- `.txt` 稿件读取。
- `.mp3` 音频路径和格式校验。
- 统一 ASR 接口。
- mock ASR 服务，用于 fixture 驱动和常规测试。
- 本地 `paraformer-zh` 服务实现，内部使用 FunASR `AutoModel`，默认模型目录为 `/Users/sakana/PyEnv/paraformer`。
- macOS MPS 推理支持，默认设备为 `mps`。
- 可通过 `--segmenter` 选择分句实现，当前内置 `regex`、`hanlp`、`jieba-subtitle`，并支持可选在线 `llm` 分句。
- 支持 `[[NO_SPLIT]]...[[/NO_SPLIT]]` 成对标记保护不分句片段。
- 支持分句单独运行，输出一行一句的可编辑分句文本，并支持从编辑后的分句文本继续执行完整流程。
- 基础文本归一化。
- 基于 `difflib.SequenceMatcher` 的顺序全局对齐。
- 基于顺序窗口 fuzzy 匹配的句子到 ASR token 匹配。
- 匹配阶段已加入轻量数字读法兼容，例如稿件 `12元` 可对齐 ASR 的「十二元」。
- 句子时间使用选中的 ASR 候选窗口完整 token 范围，替换字和数字读法差异不会造成时间轴缺口。
- 本地 Qwen3 forced aligner 服务，默认模型目录为 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`，默认 `device_map = "mps"`、`dtype = "bfloat16"`。
- `asr-fuzzy`、`qwen3-forced` 和 `hybrid` 三种时间轴策略。
- 默认配置 `configs/aligner-qwen3.toml` 使用 `hybrid`，最终时间以 Qwen3 forced alignment 为主，ASR fuzzy 分支写入 telemetry。
- 带无重叠约束的句子级时间合并。
- SRT 字幕渲染接口和 `sentence_timeline.srt` 输出。
- 丰富 JSON 中间产物和诊断报告，包括 `forced_alignment.json` 和 `telemetry.json`。
- 单元、集成和端到端测试。
- 当前 `pytest --collect-only -q` 收集 67 个测试，覆盖 mock 流程、CLI 流程、`paraformer-zh` 结果转换逻辑、forced alignment 映射、hybrid telemetry、LLM block 重试与 HanLP fallback、顺序窗口 fuzzy 匹配、时间无重叠修正、SRT 首尾对齐和后处理、保护区分句、HanLP/jieba 短字幕分句、可编辑分句输入和真实 e2e 入口。
- 已手动验证本地 `paraformer-zh` + `mps` 可完成真实音频推理，并可通过 CLI 生成完整 JSON 输出。

当前完整流程输出文件：

- `word_timeline.json`
- `manuscript_segments.json`
- `normalized_text.json`
- `alignment.json`
- `sentence_timeline.json`
- `sentence_timeline.srt`
- `subtitle_render_report.json`
- `alignment_report.json`

`qwen3-forced` 或 `hybrid` 模式还会输出：

- `forced_alignment.json`
- `telemetry.json`

示例命令、输出目录、每个输出文件的内容和 schema 见 `docs/output-artifacts.md`。

当前已固定一套真实 `paraformer-zh` ASR fuzzy 结果样例：

```text
tests/fixtures/stage1_paraformer/
```

该目录包含：

- `audio.mp3`
- `manuscript.txt`
- `word_timeline.json`
- `manuscript_segments.json`
- `normalized_text.json`
- `alignment.json`
- `sentence_timeline.json`
- `sentence_timeline.srt`
- `alignment_report.json`

该样例由本地 `/Users/sakana/PyEnv/paraformer` 模型使用 `mps` 推理生成，并已用当前 ASR fuzzy 匹配逻辑重新生成下游 JSON，可作为稳定输入。

### 已确认输入

- 稿件格式：`.txt` 纯文本。
- 稿件内容：分段文本，保留段落信息。
- 音频格式：当前完整流程要求 `.mp3`。
- 输出格式：以 JSON 为主，字段尽可能丰富，方便后续分析和调整。

### 暂不处理内容

- `.wav`、`.ogg` 等多音频格式完整支持。
- 音频自动转换为统一输入格式。
- 完整数字读法归一化，例如日期、金额、百分比、长数字和复杂单位的互相转换。当前仅在匹配阶段提供轻量兼容，用于覆盖 `12元` 与「十二元」这类常见差异。
- 领域词、产品名、缩写、同义词替换。
- `.vtt`、`.csv` 等字幕或表格导出。

### 必须预留的扩展点

- ASR 服务统一接口：每个真实 ASR 服务都实现同一接口。
- Forced alignment 服务独立于 ASR 接口：该服务接收音频和真实文本，输出文本单元时间戳，不能伪装成 `AsrService`。
- 句子切分统一接口：当前内置 `regex`、`hanlp`、`jieba-subtitle` 和 `llm`，后续可继续增加实现。
- 音频输入适配层：当前接收 `.mp3`，后续扩展多格式检测和格式转换。
- 渲染输出统一接口：当前已实现 SRT，后续可扩展 VTT、CSV 等格式。
- 输出 schema 可扩展：保留中间字段和诊断字段，避免后续分析缺少排查依据。

## 建议代码结构

```text
src/
  funasr_timeline/
    __init__.py
    asr/
      base.py
      paraformer_zh_service.py
      mock_service.py
    forced_alignment/
      __init__.py
      base.py
      config.py
      factory.py
      mock_service.py
      qwen3_service.py
      sentence_mapper.py
    segmentation/
      __init__.py
      base.py
      factory.py
      editable.py
      normalization.py
      protection.py
      regex.py
      hanlp.py
      jieba_subtitle.py
      llm.py
    render/
      __init__.py
      base.py
      srt.py
    manuscript.py
    normalization.py
    alignment.py
    sentence_matching.py
    merge.py
    report.py
    cli.py
tests/
  fixtures/
  unit/
  integration/
  e2e/
```

结构可按实现需要微调，但应保持模块职责清晰：

- `asr/base.py` 定义统一 ASR 接口和标准时间轴数据结构。
- `asr/paraformer_zh_service.py` 放本地 `paraformer-zh` 模型接入实现。
- `asr/mock_service.py` 用于测试和可重复验证。
- `forced_alignment/base.py` 定义 forced alignment 接口和标准 forced unit 数据结构。
- `forced_alignment/config.py` 负责读取 aligner TOML 配置。
- `forced_alignment/qwen3_service.py` 放本地 Qwen3 forced aligner 接入实现。
- `forced_alignment/mock_service.py` 用于常规测试避免加载真实 Qwen3 模型。
- `forced_alignment/sentence_mapper.py` 负责把 forced units 映射回稿件分句时间范围。
- `manuscript.py` 负责 `.txt` 稿件读取和基础元数据。
- `segmentation/base.py` 定义句子切分接口和标准分句数据结构。
- `segmentation/factory.py` 负责分句实现注册和创建。
- `segmentation/regex.py`、`segmentation/jieba_subtitle.py`、`segmentation/llm.py` 分别放具体分句实现。
- `segmentation/protection.py`、`segmentation/editable.py`、`segmentation/normalization.py` 放保护段、可编辑分句和归一化范围附加逻辑。
- `normalization.py` 负责基础归一化。
- `alignment.py` 负责顺序全局对齐。
- `sentence_matching.py` 负责 ASR fuzzy 的顺序窗口句子匹配。
- `merge.py` 负责句子级时间合并。
- `render/base.py` 负责句子时间轴渲染接口，`render/srt.py` 负责 SRT 输出。
- `report.py` 负责诊断报告生成。
- `cli.py` 负责命令行入口。

## 流程设计

### 1. 输入读取

命令行接收显式路径，不依赖固定文件名：

- `--manuscript path/to/input.txt`
- `--audio path/to/input.mp3`
- `--output-dir path/to/output`
- `--segmenter regex|hanlp|jieba-subtitle|llm`
- `--segment-threshold 10`
- `--llm-config configs/llm-siliconflow.toml`
- `--segment-only`
- `--segments path/to/editable_segments.txt`
- `--subtitle-gap-threshold-ms 67`
- `--subtitle-min-duration-ms 200`
- `--timeline-provider asr-fuzzy|qwen3-forced|hybrid`
- `--aligner-config configs/aligner-qwen3.toml`
- `--asr-provider mock|paraformer-zh`
- `--mock-word-timeline tests/fixtures/word_timeline.json`
- `--paraformer-model-dir /Users/sakana/PyEnv/paraformer`
- `--paraformer-device mps`

后续阶段可以增加：

- `--audio-format`
- `--converted-audio`

### 2. ASR 字符时间轴

通过统一 ASR 接口生成标准化时间轴。接口不应把调用方绑定到某个具体 ASR 服务。

当前标准输出：

```json
{
  "audio": {
    "path": "input.mp3",
    "format": "mp3",
    "duration_ms": 123456
  },
  "asr": {
    "provider": "paraformer-zh",
    "model": "paraformer-zh:/Users/sakana/PyEnv/paraformer",
    "text": "识别文本"
  },
  "tokens": [
    {
      "index": 0,
      "text": "识",
      "start_ms": 880,
      "end_ms": 1120,
      "confidence": null,
      "source": "paraformer-zh"
    }
  ]
}
```

当前已实现两种 ASR 服务：

- `mock`：读取 fixture `word_timeline.json`，用于常规测试。
- `paraformer-zh`：使用本地 `paraformer-zh` 模型目录，通过 FunASR `AutoModel` 推理，默认设备为 `mps`。代码位于 `src/funasr_timeline/asr/paraformer_zh_service.py`。

当前本地 `paraformer-zh` 推理会读取 FunASR 返回的 `text` 和 `timestamp` 字段，将去空白后的文本与 timestamp 对齐后生成项目标准 token 时间轴。如果 ASR 文本包含标点但 timestamp 不包含标点，会忽略标点后再匹配；如果 FunASR 对少量连续英文或数字片段只返回一个 timestamp，会将该片段合并为多字符 ASR token。下游对齐会继续展开 token 文本，因此最终仍以稿件原文为准。

后续可继续接入或组合：

- `fsmn-vad`
- `ct-punc`

### 3. 稿件句子切分

句子切分通过 `SentenceSegmenter` 接口实现，当前内置 `regex`、`hanlp`、`jieba-subtitle` 和可选在线 `llm` 四个实现。

默认强边界：

- `。`
- `！`
- `？`
- `；`
- 段落换行

`regex` 实现不按逗号切长句，适合保留自然句。`jieba-subtitle` 实现会先按强标点和段落形成基础范围，再把逗号、顿号等标点作为短语边界，去除标点后使用 jieba 分词拼接短句，默认目标为单句归一化文本不超过 10 个字符，并保证同一个词不会被切到两个句子中。该实现还内置少量短视频字幕常见软切分点，例如 `特别`、`直接`，用于把过长表达拆成更自然的语义短语。

`hanlp` 实现使用中文多任务模型执行 constituency 解析。输入先按 `，,、。！？!?；;：:` phrase 边界拆句，保证 fallback 不跨逗号、顿号、句号、分号或冒号，再从语法树叶节点取得 token 并按顺序窗口拼接。加入下一个 token 会超过折算阈值时切分，默认阈值为 10；汉字计 1，英文和数字每两个计 1，标点、空白、符号和分隔符不计。单个 token 即使超过阈值也保持完整。模型由 HanLP 下载并复用本地缓存。

`llm` 实现通过 TOML 配置读取 OpenAI-compatible Chat Completions 端点、模型、超时和 API key 环境变量。prompt 使用 Jinja2 渲染，包含任务说明、短视频字幕约束、保护段说明、纯文本换行输出约束和多样化 few-shot 示例，并把待分割文本放在最后。进入 LLM 前先由确定性预处理移除保护段，保护标记两侧形成强制 block 边界，紧邻标记的外侧边界标点从模型视图中裁掉。每个预处理 block 分别创建一个只包含该 block 的 LLM 请求，所有 block 以 block 数量作为并发数同时执行，不再要求模型输出跨 block 分隔符。

每个 block 独立维护请求、校验和反馈重试状态。覆盖校验会从原文和模型结果中删除标点、空格、换行及其他分隔符，对剩余字母、中文和数字执行 Unicode 归一化后比较；模型新增空格或改变标点可以接受，漏字、改字和数字变化仍会失败。校验通过后立即结束该 block 的任务，不再参与后续反馈；失败 block 只携带自己的原文、上次输出和错误独立重试。程序根据通过校验的内容字符顺序定位原稿 offset，从原稿切片生成最终文本，因此模型新增的空格和标点不会进入字幕。所有 block 完成后按初始顺序合并，再按规则去掉分句两端的边界标点并把保护段按原顺序和 offset 拼回。

prompt 会强调常规口播字幕的折算长度优先控制在 4 到 8 个汉字，硬上限为 10；汉字计 1，英文和数字每两个计 1，标点符号不计。请求异常、响应解析错误和内容校验错误都在当前 block 内独立重试。重试耗尽后默认只记录 error 日志，并把失败 block 交给可配置 fallback 分句器，默认使用 `hanlp`；`--llm-raise-on-error` 可改为直接抛错并终止。成功 block 不重跑，最终仍按原 block 顺序合并。`manuscript_segments.json` 为每个分句记录 `segmenter` 和 `source_block_id`，诊断文件记录失败原因与最终 block 策略，供后续 telemetry 使用。

稿件中可以用成对标记保护不需要自动分句的片段：

```text
普通句子。[[NO_SPLIT]]这部分。不要切开！整体保留。[[/NO_SPLIT]]继续分句。
```

标记由上游直接写入输入 `.txt` 稿件，不在分句阶段按关键词动态推断。标记本身会在进入归一化、匹配和字幕输出前移除；标记包住的内容会作为一个 `boundary` 为 `protected` 的完整分句。开始和结束标记位置都属于强制分句边界，普通文本不能跨保护段合并。保护段不发送给 LLM，模型处理完成后由程序按原位置拼回；缺少结束标记、孤立结束标记和嵌套标记均视为输入错误。同一输入用于 TTS 时，先通过 `remove_no_split_markers()` 生成仅移除控制标记的朗读文本。真实剪映 E2E 的 `DEMO_TEXT` 直接包含保护标记，并分别验证 TTS 输入、LLM 请求、`manuscript_segments.json` 和最终字幕中的保护段行为。

分句也可以单独运行：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --output-dir path/to/segments \
  --segment-only \
  --segmenter jieba-subtitle
```

该命令输出 `editable_segments.txt` 和 `manuscript_segments.json`。`editable_segments.txt` 为一行一句、空行分段的人工可编辑格式。编辑后可以通过 `--segments path/to/editable_segments.txt` 替代自动分句结果继续执行 ASR 后的匹配、合并和渲染流程。

建议句子结构：

```json
{
  "index": 0,
  "text": "第一句话。",
  "paragraph_index": 0,
  "char_start": 0,
  "char_end": 5,
  "boundary": "punctuation",
  "segmenter": "regex",
  "source_block_id": null,
  "normalized_text": "第一句话",
  "normalized_start": 0,
  "normalized_end": 4
}
```

### 4. 文本归一化

当前只做基础归一化：

- 去除用于对齐的标点。
- 去除或压缩空白。
- 全角和半角字符归一。
- 英文转小写。
- 保留原始字符到归一化字符的 offset 映射。

当前匹配阶段额外提供轻量数字读法兼容，例如 `12元` 可以和 ASR 的「十二元」对齐。该逻辑只影响匹配和时间窗口选择，不改变最终字幕文本。

暂不处理：

- 日期、百分比、复杂金额、长数字、复杂单位归一化。
- 领域词、同义词和专名替换。

### 5. 顺序全局对齐

对齐对象：

- 归一化后的稿件全文。
- 归一化后的 ASR 全文。

当前使用 Python 标准库 `difflib.SequenceMatcher` 建立可解释的顺序对齐结果。后续如边界精度不足，可替换为自定义动态规划编辑距离对齐或引入 RapidFuzz 辅助诊断。

对齐结果应至少包含：

- 稿件归一化字符 offset 到 ASR token index 的映射。
- 完整 opcodes 或等价编辑操作。
- 未匹配稿件字符。
- 未映射 ASR 字符。
- 每个句子的局部匹配统计。

### 6. 句子时间合并

对每个稿件句子执行：

1. 获取句子在归一化稿件全文中的字符范围。
2. 查询该范围内映射到的 ASR token。
3. 使用第一个映射 token 的 `start_ms` 作为句子开始时间。
4. 使用最后一个映射 token 的 `end_ms` 作为句子结束时间。
5. 计算匹配分数和诊断状态。
6. 对低置信度或缺失时间的句子保留详细诊断，不静默丢弃。

建议 `sentence_timeline.json` 尽量包含丰富字段：

```json
[
  {
    "index": 0,
    "text": "第一句话。",
    "paragraph_index": 0,
    "start_ms": 880,
    "end_ms": 5195,
    "duration_ms": 4315,
    "match_score": 0.96,
    "status": "ok",
    "matched_token_indexes": [0, 1, 2, 3],
    "normalized_text": "第一句话",
    "manuscript_char_range": [0, 5],
    "normalized_char_range": [0, 4],
    "asr_token_range": [0, 3],
    "diagnostics": {
      "matched_chars": 4,
      "unmatched_manuscript_chars": [],
      "extra_asr_tokens_nearby": []
    }
  }
]
```

### 7. 对齐诊断报告

`alignment_report.json` 应帮助人工复核，建议包含：

- 总体输入摘要。
- ASR 服务和模型摘要。
- 归一化配置。
- 句子切分配置。
- 全局匹配分数。
- 低置信度句子。
- 未匹配稿件字符。
- 未映射 ASR token。
- 大时间间隔。
- 推断边界和异常边界。
- 对齐 opcodes 或摘要。

### 8. 字幕和表格导出

当前已通过 `render/base.py` 提供渲染接口，并在 `render/srt.py` 实现 `SrtTimelineRenderer`。pipeline 会基于最终 `sentence_timeline.json` 中的句子文本和无重叠时间范围生成：

- `.srt`
- `subtitle_render_report.json`

SRT 渲染规则：

- 字幕正文使用稿件原句 `text`。
- 时间格式为 `HH:MM:SS,mmm`。
- 缺失 `start_ms` 或 `end_ms` 的句子不会渲染为字幕块。
- 渲染顺序与 `sentence_timeline.json` 中的句子顺序一致。
- 相邻有效字幕之间大于 0 且不超过 `--subtitle-gap-threshold-ms` 的空隙视为空白闪轴，默认阈值为 67ms（30fps 下 2 帧）；渲染时把上一条字幕延长到下一条开始时间。设置为 `0` 可关闭。
- 持续时间短于 `--subtitle-min-duration-ms` 的字幕默认优先向右、再向左利用空闲时间延长，默认最短时间为 200ms（30fps 下 6 帧）。修正不得产生相邻字幕重叠；空间不足时保留可达到的时间并记录为未完全修复。设置为 `0` 可关闭。
- 默认仅在 SRT 渲染时将第一条有效字幕的开始时间对齐到音频起点 `00:00:00,000`，可通过 `--no-align-first-subtitle-to-audio-start` 关闭。
- 默认将最后一条有效字幕的结束时间对齐到字幕对齐音频的实际结束时间，并向上取整到 30fps 帧边界。CLI 通过 `--subtitle-alignment-audio` 接收该音频，未指定时使用 `--audio`，并可通过 `--no-align-last-subtitle-to-audio-end` 关闭。
- 末条结束时间修正仅存在于 SRT renderer 输出，不回写 `SentenceTimelineItem`，也不改变 `sentence_timeline.json`、匹配或诊断结果。
- 闪轴修正同样只作用于渲染副本，不回写主时间轴；每次修正和无法完全修复的短字幕写入 `subtitle_render_report.json`。
- 如果 TTS 音频为适配 ASR 而转换格式，字幕对齐使用转换前的原始 TTS 音频，避免编码或容器时长差异影响末条字幕。

后续可继续增加：

- `.vtt`
- `.csv`

字幕导出必须继续以稿件原文作为显示文本。

## ASR Fuzzy 实现方案

ASR fuzzy 已在真实 `paraformer-zh` token 级时间轴可用的基础上，改进句子到 token 时间范围的匹配方式。由于当前音频主要由 ground truth 稿件文本生成，不按真实多人对话或大量口语错乱场景设计，方案保持简单、顺序、可解释。

## Qwen3 Forced Aligner 当前实现

当前已接入 `Qwen3-ForcedAligner-0.6B`，用于将真实稿件文本直接强制对齐到 TTS 音频时间窗。详细设计和 MPS 可行性结论见 `docs/forced-aligner-plan.md`。

已完成本地可行性验证：

- 本地模型目录为 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`。
- 项目 `.venv` 中已可 import `qwen_asr.Qwen3ForcedAligner`。
- macOS MPS 可用，推荐默认配置为 `device_map="mps"` 和 `dtype="bfloat16"`。
- 在 `tests/fixtures/stage1_paraformer/audio.mp3` 上已跑通一次整篇文本推理。
- `bfloat16` 加载约 2.6 秒，对齐约 3.5 秒，最大 RSS 约 866 MB。
- 模型返回文本会跳过标点，归一化后可与稿件归一化文本一致。

已确认边界：

- 第一版假设输入音频均在 5 分钟以内，不实现长音频自动分块。
- 默认时间轴策略改为 `hybrid`：同时运行 Qwen3 forced aligner 和 `paraformer-zh` ASR fuzzy。
- `hybrid` 模式下 forced aligner 作为 primary 时间来源，ASR fuzzy 结果进入 telemetry，用于后续置信度分析。
- CLI 应读取 aligner 配置文件，并根据 `timeline-provider` 选择 `asr-fuzzy`、`qwen3-forced` 或 `hybrid`。
- 输出应新增 `forced_alignment.json` 和 `telemetry.json`，并在 `alignment_report.json` 中保留 telemetry 摘要。

### 当前时间轴边界

当前实现继续遵守以下约束：

- 分句通过统一接口选择，当前可用 `regex`、`hanlp`、`jieba-subtitle` 和 `llm`；CLI 默认值为 `regex`。
- 最终句子文本仍以 ground truth 稿件原文为准。
- 不处理领域词、同义词等复杂归一化。数字读法目前只做轻量匹配兼容，不作为完整归一化能力。
- 不引入全局动态规划或复杂搜索，除非后续 fixture 证明确有必要。
- 最终 `sentence_timeline.json` 中相邻句子的时间范围必须不重叠。

### 分句方案

分句继续通过 `segmentation/base.py` 的可替换接口实现。

默认规则：

- 按段落边界切分并保留 `paragraph_index`。
- 段落内部按强标点切分。
- 强边界包括 `。`、`！`、`？`、`!`、`?`、`；`、`;`。
- `regex` 输出保留强边界标点；`hanlp` 将 phrase 标点作为强制切分点并从输出两端移除；`jieba-subtitle` 输出短字幕风格文本，不保留边界标点；`llm` 原始输出保留标点用于校验，最终分句再去掉首尾边界标点。
- 不按逗号切分。
- 空段落不产出句子。

句子结构继续使用当前 `SentenceSegment` 字段：

```json
{
  "index": 0,
  "text": "原始句子。",
  "paragraph_index": 0,
  "char_start": 0,
  "char_end": 6,
  "boundary": "punctuation",
  "normalized_text": "原始句子",
  "normalized_start": 0,
  "normalized_end": 4
}
```

### 顺序窗口 fuzzy 匹配

`sentence_matching.py` 用于把每个稿件句子按顺序匹配到 ASR token 时间轴上。匹配不在整条 ASR 文本中自由搜索，而是从上一句结束 token 之后开始，保证整体顺序单调。

单句匹配步骤：

1. 取当前句子的 `normalized_text`。
2. 从当前 ASR cursor 开始构造候选窗口。
3. 根据句子归一化长度 `N` 限制候选长度，建议范围为 `N * 0.6` 到 `N * 1.5`。
4. 搜索上限建议为从 cursor 起向后最多 `N * 2.5 + 常数` 个 token。
5. 对每个候选窗口拼接归一化 ASR 文本，用 `difflib.SequenceMatcher(..., autojunk=False).ratio()` 计算相似度。
6. 选择分数最高的候选作为本句匹配结果。
7. 在候选窗口内部再做字符级对齐，用于诊断完全匹配字符；最终时间范围使用选中的 ASR 候选窗口完整 token 范围，避免替换字或数字读法差异导致时间轴缺口。

匹配输出建议使用独立数据结构，例如：

```json
{
  "sentence_index": 0,
  "match_score": 0.96,
  "matched_token_indexes": [0, 1, 2, 3],
  "asr_token_range": [0, 3],
  "matched_asr_text": "原始句子",
  "candidate_count": 8,
  "selected_candidate_rank": 1,
  "status": "ok",
  "diagnostics": {
    "text_similarity": 0.96,
    "matched_chars": 4,
    "total_normalized_chars": 4,
    "unmatched_manuscript_chars": []
  }
}
```

低置信度阈值沿用可配置参数，初始默认可继续使用 `0.8`。低于阈值时不静默丢弃，应输出 `low_confidence` 并保留候选和未匹配字符诊断。

### 时间合并与无重叠约束

`merge.py` 根据 `sentence_matching.py` 的匹配结果合并 ASR fuzzy 时间，而不是只依赖全文 alignment 中的字符范围映射。时间合并使用选中 ASR 候选窗口的完整 token 范围；字符级完全匹配 token 仅作为 diagnostics 保留。

原始时间计算：

1. 若存在匹配 token，取第一个匹配 token 的 `start_ms` 作为 `raw_start_ms`。
2. 取最后一个匹配 token 的 `end_ms` 作为 `raw_end_ms`。
3. 若没有匹配 token，则 `raw_start_ms`、`raw_end_ms`、`start_ms`、`end_ms` 均为 `null`。

最终时间必须按句子顺序消除重叠：

1. 按 `index` 从小到大遍历句子。
2. 若当前句 `raw_start_ms < previous_end_ms`，则将最终 `start_ms` 调整为 `previous_end_ms`。
3. 若当前句 `raw_end_ms < start_ms`，则标记为 `invalid_time_range`，并保留原始时间用于排查。
4. 若相邻句之间存在自然空隙，保留空隙，不强行拉齐。
5. 正常情况下只调整当前句，不回写上一句时间。

当前 `sentence_timeline.json` 已包含字段：

```json
{
  "start_ms": 1200,
  "end_ms": 2500,
  "duration_ms": 1300,
  "raw_start_ms": 1180,
  "raw_end_ms": 2500,
  "time_adjusted": true,
  "matched_asr_text": "原始句子"
}
```

状态建议控制在少量可解释值：

- `ok`
- `low_confidence`
- `no_match`
- `empty_after_normalization`
- `invalid_time_range`

`time_adjusted` 作为独立布尔字段，不覆盖 `status`，因为一句话可以同时是 `ok` 且经过无重叠时间修正。

### ASR Fuzzy 测试

当前测试覆盖以下 ASR fuzzy 行为：

- regex 分句在段落和强标点上的稳定行为。
- 正常多句稿件能按顺序匹配到 token timeline。
- ASR 开头存在额外 token 时，第一句仍能从正确内容开始取时间。
- ASR 局部少字或错字时，句子保留低置信度诊断。
- 相邻句子的原始匹配时间存在重叠时，最终 `start_ms` 和 `end_ms` 不重叠。
- 使用 `tests/fixtures/stage1_paraformer/` 作为真实 ASR fuzzy 样例，验证输入输出路径稳定。

## 测试计划

实现应覆盖三个层级。

### 单元测试

覆盖纯函数和边界情况：

- `.txt` 稿件读取。
- regex 句子切分。
- 段落换行保留。
- 标点和空白归一化。
- 全角和半角归一化。
- 英文大小写归一化。
- 原始 offset 到归一化 offset 的映射。
- ASR token 时间轴 schema 转换。
- 字符 offset 到 token index 映射。
- 句子时间合并。
- 匹配分数计算。
- ASR fuzzy 顺序窗口匹配。
- 句子时间无重叠修正。
- SRT 时间戳格式化和字幕渲染。

### 集成测试

使用小型 fixture 稿件和 fixture ASR token 时间轴测试：

- 稿件文本到句子和归一化范围的转换。
- 归一化稿件与归一化 ASR 的全局对齐。
- 替换、漏字、额外口头词等情况下的诊断。
- `sentence_timeline.json` 生成。
- `alignment_report.json` 生成。
- 基于 `tests/fixtures/stage1_paraformer/` 的 ASR fuzzy 匹配输入样例。

常规集成测试不加载真实 `paraformer-zh` 模型，避免依赖本地模型目录和长时间推理。

### 端到端测试

提供至少一个小型端到端 fixture，覆盖命令行从输入文件到输出文件的流程。

当前端到端测试包含两类：

- 默认 CLI fixture 测试：使用 mock ASR、mock forced aligner 和小型 fixture，覆盖 CLI 从输入文件到 JSON/SRT 输出的流程。
- 真实 demo 测试：复用剪映 demo 的长中文混合文本，调用剪映 TTS 生成音频，使用 LLM 分句，通过 `hybrid` 同时运行 Qwen3 forced aligner 和本地 `paraformer-zh`/FunASR ASR，并写出 `e2e_diagnostics.json`，用于检查句子数量、状态分布、telemetry 差异、TTS/音频转换/剪映草稿信息和报告摘要。

真实 demo 测试位于 `tests/e2e/test_jianying_smartsplit_demo.py`，默认会随 pytest 执行。运行前需要准备 `configs/llm-siliconflow.toml`、`configs/aligner-qwen3.toml`、可 import 的剪映 Python 接口、TTS 后端、本地模型和 `FUNASR_TIMELINE_LLM_API_KEY`。子进程 stdout/stderr 使用双线程 tee 读取，避免管道阻塞；运行 `pytest -s` 时日志实时显示，同时逐行保留到成功或失败 diagnostics。

只运行常规确定性测试时，可使用：

```bash
uv run pytest -m 'not e2e_real'
```

## 质量门禁

实现完成前应保持以下命令通过：

- 单元测试。
- 集成测试。
- 端到端测试。
- lint 检查。
- format 检查。
- typecheck 检查。

推荐工具：

- `pytest`
- `ruff`
- `mypy`

最终选择必须反映在 `pyproject.toml` 和本文档中。

## 文档同步要求

任何行为变化都必须在同一轮变更中更新相关文档：

- 流程步骤、输出 schema 或质量门禁变化时，更新本文档。
- 模型选择、外部参考或 ASR 接入策略变化时，更新 `docs/research-summary.md`。
- 命令行参数和文件布局确定后，补充使用文档。
- JSON 字段变化时，更新示例和 schema 描述。

## 已确认决策

- 稿件使用 `.txt` 纯文本，内容为分段文本。
- 当前先支持 `.mp3` 音频。
- 后续需要支持 `.wav`、`.ogg` 等常见格式，或通过转换统一输入格式。
- 句子切分保留接口，当前内置 `regex`、`hanlp`、`jieba-subtitle` 和 `llm`。
- JSON 输出要尽可能丰富，服务于后续分析调整。
- 完整数字归一化、领域词、同义词等暂不处理；当前仅保留轻量数字读法匹配兼容。
- 真实 ASR 模型通过统一接口接入；当前已实现本地 `paraformer-zh` 服务，内部使用 FunASR `AutoModel`。
- Forced alignment 通过独立接口接入；当前已实现本地 `qwen3-forced` 和测试用 `mock`。
- 本地 `paraformer-zh` 模型目录为 `/Users/sakana/PyEnv/paraformer`。
- 本地 Qwen3 forced aligner 模型目录为 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`。
- macOS 上默认使用 `mps` 推理。
- 最终句子文本默认使用原始稿件。
- CLI `--segmenter` 默认值为 `regex`；短视频字幕可显式使用 `hanlp`、`jieba-subtitle` 或 `llm`。
- 默认 aligner 配置为 `hybrid`，主时间来源为 `qwen3-forced`，ASR fuzzy 结果保留到 diagnostics 和 telemetry。
- ASR fuzzy 句子到 token 的匹配采用顺序窗口 fuzzy 匹配。
- 最终句子时间范围必须保证相邻句子不重叠。
- ASR fuzzy 句子时间使用选中 ASR 候选窗口的完整 token 范围，字符级完全匹配结果只作为诊断字段保留。
- 当前已通过 render 接口输出 `sentence_timeline.srt`。

## 待后续确认

- `.mp3` 解码依赖和本地环境要求。
- 多音频格式转换使用 `ffmpeg` 还是其他库。
- 低置信度阈值是否需要暴露到 CLI 或配置文件。
- 是否将真实 `paraformer-zh` 推理纳入可选慢速测试。
- 是否增加 `.vtt` 或 `.csv`。
- hybrid 后续是否根据 telemetry 做自动仲裁或兜底。
