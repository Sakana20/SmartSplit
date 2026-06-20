# 实现计划

## 项目目标

本项目要实现一个可检查、可扩展的音频时间轴生成流程：

1. 从配音音频中生成字符级或 token 级 ASR 时间轴。
2. 将 ASR 文本与已有稿件纯文本进行顺序对齐。
3. 以稿件原文为准，将字符级时间范围合并为句子级时间范围。
4. 输出丰富的 JSON 中间产物和诊断信息，供后续阶段调参、复核和扩展。

第一阶段重点是跑通 `.txt` 稿件和 `.mp3` 音频的主流程，并为后续多音频格式、音频转换、更多 ASR 服务扩展和字幕导出留下清晰接口。当前真实模型实现为本地 `paraformer-zh`，通过 FunASR `AutoModel` 加载。

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

## 第一阶段范围

第一阶段实现最小可用但信息丰富的离线流程。

### 实现状态

当前第一阶段已实现：

- `pyproject.toml` 驱动的 `uv` 项目结构。
- `funasr-timeline` 命令行入口。
- `.txt` 稿件读取。
- `.mp3` 音频路径和格式校验。
- 统一 ASR 接口。
- mock ASR 服务，用于 fixture 驱动和常规测试。
- 本地 `paraformer-zh` 服务实现，内部使用 FunASR `AutoModel`，默认模型目录为 `/Users/sakana/PyEnv/paraformer`。
- macOS MPS 推理支持，默认设备为 `mps`。
- 可通过 `--segmenter` 选择分句实现，当前内置 `regex`、`jieba-subtitle`，并支持可选在线 `llm` 分句。
- 支持 `[[NO_SPLIT]]...[[/NO_SPLIT]]` 成对标记保护不分句片段。
- 支持分句单独运行，输出一行一句的可编辑分句文本，并支持从编辑后的分句文本继续执行完整流程。
- 基础文本归一化。
- 基于 `difflib.SequenceMatcher` 的顺序全局对齐。
- 基于顺序窗口 fuzzy 匹配的句子到 ASR token 匹配。
- 匹配阶段已加入轻量数字读法兼容，例如稿件 `12元` 可对齐 ASR 的「十二元」。
- 句子时间使用选中的 ASR 候选窗口完整 token 范围，替换字和数字读法差异不会造成时间轴缺口。
- 带无重叠约束的句子级时间合并。
- SRT 字幕渲染接口和 `sentence_timeline.srt` 输出。
- 丰富 JSON 中间产物和诊断报告。
- 单元、集成和端到端测试。
- 当前测试数量为 24 个，覆盖 mock 流程、CLI 流程、`paraformer-zh` 结果转换逻辑、第二阶段 fuzzy 匹配、时间无重叠修正、SRT 渲染、保护区分句、jieba 短字幕分句、可编辑分句输入和按 ASR 候选窗口落时间轴。
- 已手动验证本地 `paraformer-zh` + `mps` 可完成真实音频推理，并可通过 CLI 生成完整 JSON 输出。

当前完整流程输出文件：

- `word_timeline.json`
- `manuscript_segments.json`
- `normalized_text.json`
- `alignment.json`
- `sentence_timeline.json`
- `sentence_timeline.srt`
- `alignment_report.json`

示例命令、输出目录、每个输出文件的内容和 schema 见 `docs/output-artifacts.md`。

当前已固定一套真实 `paraformer-zh` 第一阶段结果样例：

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

该样例由本地 `/Users/sakana/PyEnv/paraformer` 模型使用 `mps` 推理生成，并已用当前第二阶段逻辑重新生成下游 JSON，可作为第二阶段匹配逻辑的稳定输入。

### 已确认输入

- 稿件格式：`.txt` 纯文本。
- 稿件内容：分段文本，保留段落信息。
- 音频格式：第一阶段先跑通 `.mp3`。
- 输出格式：以 JSON 为主，字段尽可能丰富，方便第二阶段分析和调整。

### 暂不处理内容

- `.wav`、`.ogg` 等多音频格式完整支持。
- 音频自动转换为统一输入格式。
- 完整数字读法归一化，例如日期、金额、百分比、长数字和复杂单位的互相转换。当前仅在匹配阶段提供轻量兼容，用于覆盖 `12元` 与「十二元」这类常见差异。
- 领域词、产品名、缩写、同义词替换。
- `.vtt`、`.csv` 等字幕或表格导出。

### 必须预留的扩展点

- ASR 服务统一接口：每个真实 ASR 服务都实现同一接口。
- Forced alignment 服务独立于 ASR 接口：该服务接收音频和真实文本，输出文本单元时间戳，不能伪装成 `AsrService`。
- 句子切分统一接口：当前内置 `regex`、`jieba-subtitle` 和 `llm`，后续可继续增加实现。
- 音频输入适配层：第一阶段接收 `.mp3`，后续扩展多格式检测和格式转换。
- 渲染输出统一接口：当前已实现 SRT，后续可扩展 VTT、CSV 等格式。
- 输出 schema 可扩展：保留中间字段和诊断字段，避免第二阶段缺少排查依据。

## 建议代码结构

```text
src/
  funasr_timeline/
    __init__.py
    asr/
      base.py
      paraformer_zh_service.py
      mock_service.py
    segmentation/
      __init__.py
      base.py
      factory.py
      editable.py
      normalization.py
      protection.py
      regex.py
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
- `asr/mock_service.py` 用于测试和第一阶段可重复验证。
- `manuscript.py` 负责 `.txt` 稿件读取和基础元数据。
- `segmentation/base.py` 定义句子切分接口和标准分句数据结构。
- `segmentation/factory.py` 负责分句实现注册和创建。
- `segmentation/regex.py`、`segmentation/jieba_subtitle.py`、`segmentation/llm.py` 分别放具体分句实现。
- `segmentation/protection.py`、`segmentation/editable.py`、`segmentation/normalization.py` 放保护段、可编辑分句和归一化范围附加逻辑。
- `normalization.py` 负责基础归一化。
- `alignment.py` 负责顺序全局对齐。
- `sentence_matching.py` 负责第二阶段的顺序窗口 fuzzy 句子匹配。
- `merge.py` 负责句子级时间合并。
- `render/base.py` 负责句子时间轴渲染接口，`render/srt.py` 负责 SRT 输出。
- `report.py` 负责诊断报告生成。
- `cli.py` 负责命令行入口。

## 流程设计

### 1. 输入读取

第一阶段命令行接收显式路径，不依赖固定文件名：

- `--manuscript path/to/input.txt`
- `--audio path/to/input.mp3`
- `--output-dir path/to/output`
- `--segmenter regex|jieba-subtitle|llm`
- `--llm-config configs/llm-siliconflow.toml`
- `--segment-only`
- `--segments path/to/editable_segments.txt`
- `--asr-provider mock|paraformer-zh`
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

句子切分通过 `SentenceSegmenter` 接口实现，当前内置 `regex`、`jieba-subtitle` 和可选在线 `llm` 三个实现。

默认强边界：

- `。`
- `！`
- `？`
- `；`
- 段落换行

`regex` 实现不按逗号切长句，适合保留自然句。`jieba-subtitle` 实现会先按强标点和段落形成基础范围，再把逗号、顿号等标点作为短语边界，去除标点后使用 jieba 分词拼接短句，默认目标为单句归一化文本不超过 10 个字符，并保证同一个词不会被切到两个句子中。该实现还内置少量短视频字幕常见软切分点，例如 `特别`、`直接`，用于把过长表达拆成更自然的语义短语。

`llm` 实现通过 TOML 配置读取 OpenAI-compatible Chat Completions 端点、模型、超时和 API key 环境变量。prompt 使用 Jinja2 渲染，包含任务说明、短视频字幕约束、保护段说明、结构化 XML 输出约束和多样化 few-shot 示例，并把待分割文本放在最后。一次请求会提交多个文本段，模型必须按 `<block id="...">` 独立返回每段 `<segment>`，且输出 block id 必须与输入 input block id 完全一致；程序优先 XML 解析，失败时回退正则解析。LLM 输出必须保留原文标点，同一 block 内所有 segment 直接拼接后必须与原文完全一致。校验通过后，程序从原稿切片生成最终分句文本，并按规则去掉分句两端的边界标点。改写正文、新增空格、删掉标点或漏掉正文会直接报错，不自动回退。完整流程和独立分句会在输出目录写出 `llm_segmentation_diagnostics.json`，记录请求 block、原始响应、解析结果和失败覆盖诊断。

稿件中可以用成对标记保护不需要自动分句的片段：

```text
普通句子。[[NO_SPLIT]]这部分。不要切开！整体保留。[[/NO_SPLIT]]继续分句。
```

标记本身会在进入归一化、匹配和字幕输出前移除；标记包住的内容会作为一个 `boundary` 为 `protected` 的完整分句。

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
  "normalized_text": "第一句话",
  "normalized_start": 0,
  "normalized_end": 4
}
```

### 4. 文本归一化

第一阶段只做基础归一化：

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

第一阶段可使用 Python 标准库 `difflib.SequenceMatcher` 建立可解释的顺序对齐结果。后续如边界精度不足，可替换为自定义动态规划编辑距离对齐或引入 RapidFuzz 辅助诊断。

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

SRT 渲染规则：

- 字幕正文使用稿件原句 `text`。
- 时间格式为 `HH:MM:SS,mmm`。
- 缺失 `start_ms` 或 `end_ms` 的句子不会渲染为字幕块。
- 渲染顺序与 `sentence_timeline.json` 中的句子顺序一致。

后续可继续增加：

- `.vtt`
- `.csv`

字幕导出必须继续以稿件原文作为显示文本。

## 第二阶段实现方案

第二阶段已在第一阶段真实 `paraformer-zh` token 级时间轴可用的基础上，改进句子到 token 时间范围的匹配方式。由于当前音频主要由 ground truth 稿件文本生成，不按真实多人对话或大量口语错乱场景设计，方案保持简单、顺序、可解释。

## Qwen3 Forced Aligner 下一阶段方案

下一阶段计划接入 `Qwen3-ForcedAligner-0.6B`，用于将真实稿件文本直接强制对齐到 TTS 音频时间窗。详细方案见 `docs/forced-aligner-plan.md`。

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

### 第二阶段边界

第二阶段继续遵守以下约束：

- 分句仍采用简单的“段落 + 标点符号 regex”方案。
- 最终句子文本仍以 ground truth 稿件原文为准。
- 不处理领域词、同义词等复杂归一化。数字读法目前只做轻量匹配兼容，不作为完整归一化能力。
- 不引入全局动态规划或复杂搜索，除非后续 fixture 证明确有必要。
- 最终 `sentence_timeline.json` 中相邻句子的时间范围必须不重叠。

### 分句方案

分句继续通过 `segmentation/base.py` 的可替换接口实现，第二阶段先不扩展复杂规则。

默认规则：

- 按段落边界切分并保留 `paragraph_index`。
- 段落内部按强标点切分。
- 强边界包括 `。`、`！`、`？`、`!`、`?`、`；`、`;`。
- 标点保留在句子原文中。
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

第二阶段新增 `sentence_matching.py`，用于把每个稿件句子按顺序匹配到 ASR token 时间轴上。匹配不在整条 ASR 文本中自由搜索，而是从上一句结束 token 之后开始，保证整体顺序单调。

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

`merge.py` 在第二阶段根据 `sentence_matching.py` 的匹配结果合并时间，而不是只依赖全文 alignment 中的字符范围映射。时间合并使用选中 ASR 候选窗口的完整 token 范围；字符级完全匹配 token 仅作为 diagnostics 保留。

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

第二阶段 `sentence_timeline.json` 已新增字段：

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

### 第二阶段测试

第二阶段实现需要补充以下测试：

- regex 分句在段落和强标点上的稳定行为。
- 正常多句稿件能按顺序匹配到 token timeline。
- ASR 开头存在额外 token 时，第一句仍能从正确内容开始取时间。
- ASR 局部少字或错字时，句子保留低置信度诊断。
- 相邻句子的原始匹配时间存在重叠时，最终 `start_ms` 和 `end_ms` 不重叠。
- 使用 `tests/fixtures/stage1_paraformer/` 作为真实第一阶段样例，验证第二阶段输入输出路径稳定。

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
- 第二阶段顺序窗口 fuzzy 匹配。
- 句子时间无重叠修正。
- SRT 时间戳格式化和字幕渲染。

### 集成测试

使用小型 fixture 稿件和 fixture ASR token 时间轴测试：

- 稿件文本到句子和归一化范围的转换。
- 归一化稿件与归一化 ASR 的全局对齐。
- 替换、漏字、额外口头词等情况下的诊断。
- `sentence_timeline.json` 生成。
- `alignment_report.json` 生成。
- 基于 `tests/fixtures/stage1_paraformer/` 的第二阶段匹配输入样例。

常规集成测试不加载真实 `paraformer-zh` 模型，避免依赖本地模型目录和长时间推理。

### 端到端测试

提供至少一个小型端到端 fixture，覆盖命令行从输入文件到输出文件的流程。

当前端到端测试包含两类：

- 默认 CLI fixture 测试：使用 mock ASR、mock forced aligner 和小型 fixture，覆盖 CLI 从输入文件到 JSON/SRT 输出的流程。
- 真实 demo 测试：复用剪映 demo 的长中文混合文本，调用剪映 TTS 生成音频，使用 LLM 分句，通过 `hybrid` 同时运行 Qwen3 forced aligner 和本地 `paraformer-zh`/FunASR ASR，并写出 `e2e_diagnostics.json`，用于检查句子数量、状态分布、telemetry 差异、TTS/音频转换/剪映草稿信息和报告摘要。

真实 demo 测试位于 `tests/e2e/test_jianying_smartsplit_demo.py`，默认会随 pytest 执行。运行前需要准备 `configs/llm-siliconflow.toml`、`configs/aligner-qwen3.toml`、可 import 的剪映 Python 接口、TTS 后端、本地模型和 `FUNASR_TIMELINE_LLM_API_KEY`。

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
- 第一阶段先跑通 `.mp3` 音频。
- 后续需要支持 `.wav`、`.ogg` 等常见格式，或通过转换统一输入格式。
- 句子切分保留接口，当前内置 `regex`、`jieba-subtitle` 和 `llm`。
- 第一阶段 JSON 输出要尽可能丰富，服务于第二阶段分析调整。
- 完整数字归一化、领域词、同义词等暂不处理；当前仅保留轻量数字读法匹配兼容。
- 真实 ASR 模型通过统一接口接入；当前已实现本地 `paraformer-zh` 服务，内部使用 FunASR `AutoModel`。
- 本地 `paraformer-zh` 模型目录为 `/Users/sakana/PyEnv/paraformer`。
- macOS 上默认使用 `mps` 推理。
- 最终句子文本默认使用原始稿件。
- 第二阶段分句继续使用简单段落和强标点 regex。
- 第二阶段句子到 token 的匹配采用顺序窗口 fuzzy 匹配。
- 第二阶段最终句子时间范围必须保证相邻句子不重叠。
- 第二阶段句子时间使用选中 ASR 候选窗口的完整 token 范围，字符级完全匹配结果只作为诊断字段保留。
- 当前已通过 render 接口输出 `sentence_timeline.srt`。

## 待后续确认

- `.mp3` 解码依赖和本地环境要求。
- 多音频格式转换使用 `ffmpeg` 还是其他库。
- 低置信度阈值的默认值。
- 输出目录中各中间文件的最终命名规范。
- 是否将真实 `paraformer-zh` 推理纳入可选慢速测试。
- 是否增加 `.vtt` 或 `.csv`。
