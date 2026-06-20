# Qwen3 Forced Aligner 接入计划

本文档记录将 Qwen3 ForcedAligner 接入当前时间轴流程的可行性结论、阶段边界和实现状态。第一版已实现 `asr-fuzzy`、`qwen3-forced` 和 `hybrid` 三种时间轴策略。

## 目标

当前 ASR fuzzy 流程已经可以把稿件分句顺序匹配到 `paraformer-zh` token 时间轴。本阶段新增基于真实稿件文本的 forced alignment：

1. 对分句后的真实文本和 TTS 音频进行强制对齐。
2. 一次推理处理整篇文本，输出字级或词级时间戳。
3. 用确定性的 offset 映射把 forced alignment 结果回填到稿件分句。
4. 默认运行 hybrid 模式，同时保留 ASR fuzzy 和 forced aligner 两部分 telemetry，用于后续置信度分析。
5. 最终字幕文本继续只来自稿件原文。

该方案面向“音频由 ground truth 稿件通过 TTS 生成”的顺序稳定场景，不优先处理多人对话、插话、错序或大段缺失。

## MPS 可行性结论

已在项目现有 `.venv` 中使用本地模型完成最小实验：

- 模型目录：`/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`
- Python 入口：`qwen_asr.Qwen3ForcedAligner`
- 设备：`device_map="mps"`
- 推荐 dtype：`torch.bfloat16`
- 测试音频：`tests/fixtures/stage1_paraformer/audio.mp3`
- 测试稿件：`tests/fixtures/stage1_paraformer/manuscript.txt`

实测结果：

- `qwen_asr` 已可在项目 `.venv` 中 import。
- `torch.backends.mps.is_available()` 为 `True`。
- `bfloat16`、`float16`、`float32` 在 MPS 上均可运行。
- `bfloat16` 表现最好，加载约 2.6 秒，对齐约 3.5 秒。
- `bfloat16` 最大 RSS 约 866 MB。
- MPS driver memory 加载后约 1.77 GB，对齐后约 2.89 GB。
- `float16` 和 `float32` 可以作为兼容选项，但当前不建议作为默认值。
- 当前不需要 quant 版本；实现中应预留 dtype 配置和后续 quantization 配置字段。

模型输出观察：

- 输入原稿包含标点时，模型返回的 item 会跳过标点。
- 返回结构包含 `text`、`start_time`、`end_time`，时间单位为秒。
- 测试稿件原文 70 字，归一化后 65 字，模型返回 65 个 item。
- 模型返回文本归一化后与项目 `normalize_text(manuscript)` 完全一致。
- 因此可以直接用 `SentenceSegment.normalized_start` 和 `normalized_end` 将 forced item 范围映射回每个分句。

测试中按现有 regex 分句切回句子后得到：

```text
0: 400ms  -> 4320ms
1: 5280ms -> 9840ms
2: 10560ms -> 12880ms
```

相邻句之间的自然停顿保留为空隙，不需要填平。

## 已确认边界

- 暂时假设所有输入音频都在 5 分钟以内。
- 第一版不实现长音频自动切块。
- 输入音频仍沿用当前 `.mp3` 第一阶段约束。
- 语言默认使用 `Chinese`，允许文本中少量混杂英文和数字。
- 最终句子文本以稿件分句原文为准，不使用 ASR 或 aligner 输出文本替换。
- 默认时间轴策略为 `hybrid`。
- `hybrid` 模式下同时运行 `paraformer-zh` ASR fuzzy 和 Qwen3 forced aligner。
- forced aligner 的时间窗作为主要候选；ASR fuzzy 结果进入 telemetry，用于后续置信度、偏差和兜底策略分析。

## 方案位置

forced aligner 不应实现现有 `AsrService` 接口，因为它需要同时接收音频和真实文本：

```text
AsrService.transcribe(audio_path) -> WordTimeline
ForcedAlignmentService.align(audio_path, text, language) -> ForcedAlignmentResult
```

已新增独立模块：

```text
src/funasr_timeline/forced_alignment/
  __init__.py
  base.py
  qwen3_service.py
  factory.py
  mock_service.py
  sentence_mapper.py
```

职责划分：

- `base.py`：定义 `ForcedAlignmentService`、`ForcedAlignmentUnit`、`ForcedAlignmentResult`。
- `qwen3_service.py`：封装 `Qwen3ForcedAligner.from_pretrained(...)` 和 `align(...)`。
- `factory.py`：根据配置创建具体 forced aligner 实现。
- `mock_service.py`：测试用 fixture 服务，默认测试不加载真实 Qwen3 模型。
- `sentence_mapper.py`：将 forced alignment units 映射到 `SentenceSegment`，生成句子级时间候选。

现有 ASR 模块继续保留：

```text
src/funasr_timeline/asr/
```

`hybrid` 模式应组合两个来源，而不是把 forced aligner 伪装成 ASR。

## 配置设计

CLI 应读取 aligner 配置文件，并根据 `timeline.provider` 选择具体实现。

已新增参数：

```text
--timeline-provider asr-fuzzy|qwen3-forced|hybrid
--aligner-config configs/aligner-qwen3.toml
```

默认值：

```text
--timeline-provider hybrid
--aligner-config configs/aligner-qwen3.toml
```

配置文件结构：

```toml
[timeline]
provider = "hybrid"
primary = "qwen3-forced"

[qwen3_forced]
provider = "qwen3-forced"
model_dir = "/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B"
device_map = "mps"
dtype = "bfloat16"
language = "Chinese"
max_audio_seconds = 300

[asr]
provider = "paraformer-zh"

[paraformer_zh]
model_dir = "/Users/sakana/PyEnv/paraformer"
device = "mps"

[telemetry]
include_forced_units = true
include_asr_tokens = true
include_sentence_comparison = true
```

配置规则：

- CLI 参数优先级高于配置文件。
- `timeline.provider = "hybrid"` 时同时创建 ASR 服务和 forced aligner 服务。
- `timeline.provider = "qwen3-forced"` 时可不运行 ASR，但仍允许用户显式打开 ASR telemetry。
- `timeline.provider = "asr-fuzzy"` 保持当前流程。
- `dtype` 初始支持 `bfloat16`、`float16`、`float32`。
- 测试配置支持 `qwen3_forced.provider = "mock"` 和 `units_path`，用于默认测试绕过真实模型。
- `quantization` 字段暂不生效，但可预留给后续 quant 模型。

## 流程设计

### `asr-fuzzy`

保持当前流程：

```text
manuscript
-> segmentation
-> paraformer/mock ASR word_timeline
-> normalized_text
-> alignment
-> sentence_matching
-> merge_sentence_timelines
-> sentence_timeline / srt / report
```

### `qwen3-forced`

使用真实文本强制对齐：

```text
manuscript
-> segmentation
-> normalized manuscript + segment ranges
-> Qwen3ForcedAligner.align(audio, text, language)
-> normalize forced items
-> map forced normalized offsets to sentence ranges
-> merge with no-overlap guard
-> sentence_timeline / srt / report
```

### `hybrid`

默认模式：

```text
manuscript
-> segmentation
-> run paraformer ASR fuzzy branch
-> run Qwen3 forced branch
-> use forced branch as primary sentence timing
-> store both branches in telemetry
-> emit final sentence_timeline / srt / report
```

第一版 hybrid 不做复杂自动仲裁。建议先保留以下数据，为后续置信度规则做准备：

- 每句 forced 时间窗。
- 每句 ASR fuzzy 时间窗。
- 两者 start/end/duration 差异。
- forced item 文本与稿件归一化文本是否完全一致。
- ASR fuzzy `match_score` 和 `status`。
- ASR 全局 alignment score。
- 句内 forced unit 数量和空时间范围。

## Offset 映射策略

Qwen3 forced aligner 返回的 item 可能是字、词或混合片段，不能假设永远一字一项。映射必须保持确定性：

1. 拼接 forced items 的 `text`。
2. 使用项目现有 `normalize_text` 得到 forced normalized text。
3. 将每个 forced item 展开成 normalized char 到 item index 的映射。
4. 比较 forced normalized text 与 manuscript normalized text。
5. 完全一致时，直接使用 `SentenceSegment.normalized_start/end` 查找首尾 forced item。
6. 不完全一致时，使用顺序 `SequenceMatcher(..., autojunk=False)` 生成诊断，不静默改写字幕文本。

异常状态建议：

- `forced_ok`
- `forced_text_mismatch`
- `forced_empty_segment`
- `forced_missing_unit`
- `forced_invalid_time_range`
- `forced_audio_too_long`

最终 `SentenceTimelineItem.status` 是否复用现有 `ok`、`low_confidence` 等值，可在实现时决定；但 forced 细分状态应进入 diagnostics 或 telemetry。

## Telemetry 设计

已新增输出：

```text
telemetry.json
forced_alignment.json
```

`forced_alignment.json` 保存 forced aligner 的标准化结果：

```json
{
  "provider": "qwen3-forced",
  "model": "/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B",
  "device_map": "mps",
  "dtype": "bfloat16",
  "language": "Chinese",
  "audio": {
    "path": "input.mp3",
    "duration_ms": 12345
  },
  "text": {
    "input": "原始输入文本",
    "normalized": "归一化输入文本",
    "forced_normalized": "模型返回归一化文本",
    "normalized_text_match": true
  },
  "units": [
    {
      "index": 0,
      "text": "正",
      "normalized_text": "正",
      "start_ms": 400,
      "end_ms": 640
    }
  ]
}
```

`telemetry.json` 汇总 hybrid 分析数据：

```json
{
  "timeline_provider": "hybrid",
  "primary": "qwen3-forced",
  "forced_alignment": {
    "provider": "qwen3-forced",
    "unit_count": 65,
    "normalized_text_match": true
  },
  "asr_fuzzy": {
    "provider": "paraformer-zh",
    "token_count": 65,
    "global_match_score": 1.0
  },
  "sentences": [
    {
      "index": 0,
      "forced": {
        "start_ms": 400,
        "end_ms": 4320,
        "status": "forced_ok",
        "unit_range": [0, 22]
      },
      "asr_fuzzy": {
        "start_ms": 410,
        "end_ms": 4330,
        "status": "ok",
        "match_score": 1.0,
        "token_range": [0, 22]
      },
      "comparison": {
        "start_delta_ms": -10,
        "end_delta_ms": -10,
        "duration_delta_ms": 0
      }
    }
  ]
}
```

`alignment_report.json` 应增加 telemetry 摘要，但不必嵌入所有原始 units，避免报告过大。

## `sentence_timeline.json` 兼容策略

最终句子级时间轴继续使用现有主要字段：

- `text`
- `start_ms`
- `end_ms`
- `duration_ms`
- `raw_start_ms`
- `raw_end_ms`
- `time_adjusted`
- `status`
- `diagnostics`

hybrid 模式下：

- `start_ms` 和 `end_ms` 默认来自 forced aligner。
- `diagnostics.timeline_provider` 为 `hybrid`。
- `diagnostics.primary_timing_source` 为 `qwen3-forced`。
- `diagnostics.forced_unit_range` 保存 forced unit 范围。
- `diagnostics.asr_fuzzy` 保存当前 ASR fuzzy 句子匹配摘要。

现有 `matched_token_indexes`、`matched_asr_text`、`asr_token_range` 字段可以继续保留为 ASR fuzzy 诊断，不作为 primary 时间来源。

## 测试状态

常规测试不加载真实 Qwen3 模型。已使用 fixture 和 mock forced aligner 覆盖：

- forced units 与稿件归一化文本完全一致时的句子映射。
- forced units 跳过标点时仍能按 `normalized_start/end` 回填分句时间。
- 英文大小写、全半角和数字保留场景。
- forced text 与稿件归一化文本不一致时输出诊断。
- forced 时间范围重叠或非法时触发无重叠修正或错误状态。
- hybrid telemetry 同时包含 forced 和 ASR fuzzy 分支。
- `timeline.provider` 不同取值对应不同流程。
- 复用剪映 demo 的长中文混合文本运行真实诊断型 e2e，覆盖 LLM 分句、剪映 TTS、Qwen3 forced aligner、本地 `paraformer-zh`/FunASR、混合中文、英文、数字、金额、重复词和较长句场景。

真实剪映 TTS、本地 Qwen3 forced aligner 和本地 `paraformer-zh` 链路已进入默认 demo e2e。运行方式和环境要求见 `docs/usage.md` 的“端到端测试”部分。

## 已完成

- forced alignment 基础数据结构、Qwen3 服务、mock 服务和 factory。
- forced units 到分句时间轴的确定性 offset 映射。
- CLI `--timeline-provider` 和 `--aligner-config`。
- 默认配置 `configs/aligner-qwen3.toml`。
- 真实剪映 TTS 端到端测试环境变量配置 `configs/jianying-e2e.env`。
- `forced_alignment.json`、`telemetry.json` 和 report telemetry 摘要。
- 单元、集成和 CLI 测试；默认测试不加载真实 Qwen3 模型。

## 后续事项

1. 增加真实音频时长探测，在运行前执行 `max_audio_seconds` 检查。
2. 根据真实样本积累 telemetry，设计 hybrid 置信度和兜底仲裁规则。
3. 评估 quant 模型接入字段和加载路径。
4. 增加可选慢速真实 Qwen3 验证，不进入默认 `pytest`。
