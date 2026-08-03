# 输出产物说明

本文档记录当前示例命令、输出目录、每个 JSON 文件的内容、schema 和 SRT 字幕产物。当前仓库固定了两类示例：

- mock ASR 示例：用于快速说明输出结构，命令输出目录可设为 `test_temp/`。
- 真实 `paraformer-zh` 示例：固定在 `tests/fixtures/stage1_paraformer/`，可作为 ASR fuzzy 匹配逻辑的输入样例。

## 示例命令

```bash
uv run funasr-timeline \
  --manuscript tests/fixtures/manuscript.txt \
  --audio tests/fixtures/audio.mp3 \
  --output-dir test_temp \
  --segmenter regex \
  --timeline-provider asr-fuzzy \
  --asr-provider mock \
  --mock-word-timeline tests/fixtures/word_timeline.json
```

该命令会读取：

- `tests/fixtures/manuscript.txt`：示例稿件。
- `tests/fixtures/audio.mp3`：示例音频路径。mock 流程不读取真实音频内容；非 MP3 输入会先由 ffmpeg 转换。
- `tests/fixtures/word_timeline.json`：mock ASR 字符级时间轴。

输出目录：

```text
test_temp/
```

`asr-fuzzy` 流程会生成 8 个 JSON 文件和 1 个 SRT 字幕文件：

- `word_timeline.json`
- `audio_conversion.json`
- `manuscript_segments.json`
- `normalized_text.json`
- `alignment.json`
- `sentence_timeline.json`
- `sentence_timeline.srt`
- `subtitle_render_report.json`
- `alignment_report.json`

`qwen3-forced` 或 `hybrid` 流程会额外生成：

- `forced_alignment.json`
- `telemetry.json`

使用当前默认真实配置运行 hybrid 的典型命令：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir path/to/output \
  --segmenter llm \
  --llm-config configs/llm-siliconflow.toml \
  --timeline-provider hybrid \
  --aligner-config configs/aligner-qwen3.toml
```

单独分句命令会生成：

- `editable_segments.txt`
- `manuscript_segments.json`

## 固定真实模型样例

真实 `paraformer-zh` ASR fuzzy 样例已固定在：

```text
tests/fixtures/stage1_paraformer/
```

该目录包含：

- `audio.mp3`：由本地模型目录示例音频转换得到的 `.mp3` 输入。
- `manuscript.txt`：与该音频匹配的稿件文本。
- `word_timeline.json`：真实 `paraformer-zh` token 时间轴。
- `manuscript_segments.json`：稿件句子切分结果。
- `normalized_text.json`：稿件和 ASR 的归一化结果。
- `alignment.json`：全文顺序对齐结果。
- `sentence_timeline.json`：句子级时间轴。
- `sentence_timeline.srt`：由句子级时间轴渲染得到的 SRT 字幕。
- `alignment_report.json`：对齐诊断报告。

生成该样例时使用的命令：

```bash
uv run funasr-timeline \
  --manuscript test_temp/real_paraformer_manuscript.txt \
  --audio test_temp/real_paraformer_example.mp3 \
  --output-dir test_temp \
  --segmenter regex \
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh \
  --paraformer-model-dir /Users/sakana/PyEnv/paraformer \
  --paraformer-device mps
```

固定后的样例关键值：

- `word_timeline.json` 中 `asr.provider` 为 `paraformer-zh`。
- `word_timeline.json` 中 token 数为 `65`。
- `sentence_timeline.json` 中句子数为 `3`。
- `sentence_timeline.srt` 中字幕块数为 `3`。
- `alignment.json` 中 `global_match_score` 为 `1.0`。
- `alignment_report.json` 中 `low_confidence_sentences` 为空。

## `word_timeline.json`

用途：保存 ASR 服务输出的标准化字符级或 token 级时间轴。mock 流程会把输入 fixture 原样转换为项目标准结构；`paraformer-zh` 流程会把 FunASR `text` 和 `timestamp` 转换为同一结构。多数中文 token 是单字符；当 FunASR 对连续英文或数字片段只返回一个 timestamp 时，token 可能包含多个字符，下游归一化和匹配会再展开该 token 文本。

Schema：

```json
{
  "audio": {
    "path": "string",
    "format": "string",
    "duration_ms": "integer | null"
  },
  "asr": {
    "provider": "string",
    "model": "string | null",
    "text": "string"
  },
  "tokens": [
    {
      "index": "integer",
      "text": "string",
      "start_ms": "integer",
      "end_ms": "integer",
      "confidence": "number | null",
      "source": "string"
    }
  ]
}
```

字段说明：

- `audio.path`：音频路径。
- `audio.format`：送入 ASR 的音频格式，当前为 `mp3`。
- `audio.duration_ms`：音频或 ASR 时间轴持续时间，未知时为 `null`。
- `asr.provider`：ASR 服务名，例如 `mock` 或 `paraformer-zh`。
- `asr.model`：模型名或 fixture 标识。
- `asr.text`：ASR 完整识别文本。
- `tokens[].index`：token 在 ASR 时间轴中的序号。
- `tokens[].text`：token 原文，当前通常为单字或单字符。
- `tokens[].start_ms` / `tokens[].end_ms`：token 时间范围，单位为毫秒。
- `tokens[].confidence`：ASR 置信度，服务未提供时为 `null`。
- `tokens[].source`：token 来源。

当前示例要点：

- `asr.provider` 为 `mock`。
- `asr.text` 为 `嗯第一句话第二段有english123`。
- 第一个 token `嗯` 是稿件外 ASR 内容，后续会在对齐报告中标为未映射 ASR token。

## `audio_conversion.json`

用途：记录 ASR 音频输入适配。MP3 输入不会转换；其他输入会通过 ffmpeg 转换到 `output-dir/audio/`。报告在后续 ASR、对齐和渲染前写出，因此后续阶段失败时仍可用于诊断。

Schema：

```json
{
  "source_audio_path": "string",
  "asr_audio_path": "string",
  "source_format": "string",
  "asr_format": "mp3",
  "converted": "boolean",
  "conversion_reused": "boolean",
  "conversion_cache_key": "string | null",
  "ffmpeg_command": "string[] | null"
}
```

非 MP3 转换文件名包含基于源路径、大小和修改时间生成的指纹；相同输入可复用已校验的缓存，不同目录下的同名音频不会映射到同一个目标文件。转换先写随机临时文件，经 ffprobe 校验后再原子替换最终缓存文件。

## `manuscript_segments.json`

用途：保存稿件句子切分结果，以及每个句子在原稿和归一化文本中的范围。

Schema：

```json
[
  {
    "index": "integer",
    "text": "string",
    "paragraph_index": "integer",
    "char_start": "integer",
    "char_end": "integer",
    "boundary": "string",
    "segmenter": "string",
    "source_block_id": "string | null",
    "normalized_text": "string",
    "normalized_start": "integer | null",
    "normalized_end": "integer | null"
  }
]
```

字段说明：

- `index`：句子序号。
- `text`：稿件原句文本，保留原始字符、标点、全角字符和空格。
- `paragraph_index`：段落序号，从 0 开始。
- `char_start` / `char_end`：句子在原始稿件字符串中的半开区间 `[start, end)`。
- `boundary`：切分边界来源。当前常见值为 `punctuation`、`paragraph`、`protected` 或 `editable`。
- `segmenter`：实际生成该分句的实现，例如 `llm`、`hanlp`、`regex`、`jieba-subtitle`、`protected` 或 `editable`。
- `source_block_id`：使用 LLM 编排时对应的 `block-N`；同一失败 block 的 fallback 分句共享该值。非 LLM 流程及保护片段为 `null`。
- `normalized_text`：用于对齐的归一化句子文本。
- `normalized_start` / `normalized_end`：句子在归一化稿件全文中的半开区间。

自动分句器会在输出前合并孤立短分句：普通分句折算长度小于 2 个汉字时优先并入同段落前一个普通分句，前一个不可用时再尝试并入后一个普通分句；该规则不跨段落，也不会吸收或拆开 `protected` 分句。

当前示例要点：

- 共切出 2 个句子。
- 第二句原文为 `第二段有Ｅｎｇｌｉｓｈ 123！`。
- 第二句归一化后为 `第二段有english123`。

## `editable_segments.txt`

用途：保存单独分句命令生成的人类可读、可编辑分句结果。

生成命令：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --output-dir test_temp \
  --segment-only \
  --segmenter jieba-subtitle
```

格式规则：

- 每个非空行表示一个分句。
- 空行表示段落分隔。
- 人工编辑时应只调整换行位置或必要文本；后续用 `--segments` 读取该文件时，会以该文件内容作为分句后的 ground truth 文本。

示例：

```text
说实话，
荔枝真的是
夏天幸福感
特别高的水果，

尤其最近
正是季节，
```

使用编辑后分句继续完整流程：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --segments test_temp/editable_segments.txt \
  --audio path/to/audio.mp3 \
  --output-dir test_temp \
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh
```

## `normalized_text.json`

用途：保存稿件和 ASR 文本的归一化结果，以及字符级 offset 映射。该文件用于检查标点、空白、全半角、英文大小写等归一化行为。

Schema：

```json
{
  "manuscript": {
    "text": "string",
    "chars": [
      {
        "normalized_index": "integer",
        "original_index": "integer",
        "original_char": "string",
        "normalized_char": "string"
      }
    ]
  },
  "asr": {
    "text": "string",
    "chars": [
      {
        "normalized_index": "integer",
        "token_index": "integer",
        "token_text": "string",
        "normalized_char": "string"
      }
    ]
  }
}
```

字段说明：

- `manuscript.text`：归一化后的稿件全文。
- `manuscript.chars[].normalized_index`：归一化字符序号。
- `manuscript.chars[].original_index`：该归一化字符对应的原稿字符位置。
- `manuscript.chars[].original_char`：原稿字符。
- `manuscript.chars[].normalized_char`：归一化字符。
- `asr.text`：归一化后的 ASR 全文。
- `asr.chars[].token_index`：该归一化字符对应的 ASR token 序号。
- `asr.chars[].token_text`：原始 ASR token 文本。

当前示例要点：

- 稿件归一化全文为 `第一句话第二段有english123`。
- ASR 归一化全文为 `嗯第一句话第二段有english123`。
- 全角 `Ｅｎｇｌｉｓｈ` 被归一化为 `english`。
- 标点、空白和换行不进入归一化文本，但通过 `original_index` 仍可追溯到原稿。

## `alignment.json`

用途：保存归一化稿件全文和归一化 ASR 全文之间的顺序全局对齐结果。

Schema：

```json
{
  "global_match_score": "number",
  "manuscript_to_token": {
    "normalized_manuscript_index": "asr_token_index"
  },
  "opcodes": [
    {
      "tag": "string",
      "manuscript_range": ["integer", "integer"],
      "asr_range": ["integer", "integer"]
    }
  ],
  "unmatched_manuscript_indexes": ["integer"],
  "unmapped_asr_indexes": ["integer"]
}
```

字段说明：

- `global_match_score`：稿件归一化字符中成功匹配到 ASR token 的比例。
- `manuscript_to_token`：稿件归一化字符 offset 到 ASR token index 的映射。JSON key 为字符串形式。
- `opcodes`：`difflib.SequenceMatcher` 风格的编辑操作摘要。
- `opcodes[].tag`：操作类型，例如 `equal`、`insert`、`delete`、`replace`。
- `opcodes[].manuscript_range`：操作对应的稿件归一化文本半开区间。
- `opcodes[].asr_range`：操作对应的 ASR 归一化文本半开区间。
- `unmatched_manuscript_indexes`：没有匹配到 ASR token 的稿件归一化字符 offset。
- `unmapped_asr_indexes`：没有映射到稿件字符的 ASR 归一化字符 offset。

当前示例要点：

- `global_match_score` 为 `1.0`。
- `opcodes` 首项为 `insert`，表示 ASR 开头多了一个稿件中不存在的 `嗯`。
- `unmapped_asr_indexes` 为 `[0]`。
- `unmatched_manuscript_indexes` 为空。

## `sentence_timeline.json`

用途：保存最终句子级时间轴。该文件是最主要的业务输出，句子文本始终来自原始稿件。

Schema：

```json
[
  {
    "index": "integer",
    "text": "string",
    "paragraph_index": "integer",
    "start_ms": "integer | null",
    "end_ms": "integer | null",
    "duration_ms": "integer | null",
    "raw_start_ms": "integer | null",
    "raw_end_ms": "integer | null",
    "time_adjusted": "boolean",
    "match_score": "number",
    "status": "string",
    "matched_token_indexes": ["integer"],
    "matched_asr_text": "string",
    "normalized_text": "string",
    "manuscript_char_range": ["integer", "integer"],
    "normalized_char_range": ["integer | null", "integer | null"],
    "asr_token_range": ["integer | null", "integer | null"],
    "diagnostics": {
      "matched_chars": "integer",
      "total_normalized_chars": "integer",
      "unmatched_manuscript_chars": [
        {
          "normalized_index": "integer",
          "char": "string"
        }
      ],
      "candidate_count": "integer",
      "selected_candidate_rank": "integer | null",
      "text_similarity": "number",
      "candidate_window": {
        "start_token_index": "integer",
        "end_token_index": "integer"
      },
      "exact_matched_token_indexes": ["integer"],
      "extra_asr_tokens_nearby": []
    }
  }
]
```

字段说明：

- `index`：句子序号。
- `text`：最终输出句子文本，来自原始稿件。
- `paragraph_index`：段落序号。
- `start_ms` / `end_ms`：句子时间范围。`asr-fuzzy` 下来自选中 ASR 候选窗口的首尾 token；`qwen3-forced` 和 `hybrid` 下默认来自 forced alignment units。
- `duration_ms`：句子持续时间。
- `raw_start_ms` / `raw_end_ms`：顺序窗口 fuzzy 匹配直接得到的原始时间范围。
- `time_adjusted`：最终时间是否因为无重叠约束被修正。
- `match_score`：句子归一化文本和候选 ASR 文本的 fuzzy 相似度。
- `status`：句子状态。当前常见值为 `ok`、`low_confidence`、`no_match`、`empty_after_normalization`、`invalid_time_range`、`forced_missing_unit`、`forced_empty_segment`、`forced_invalid_time_range`。
- `matched_token_indexes`：用于取时间的 ASR 候选窗口 token 序号，包含替换字和数字读法差异对应的 ASR token。
- `matched_asr_text`：被选中的 ASR 候选窗口归一化文本。
- `normalized_text`：句子归一化文本。
- `manuscript_char_range`：句子在原稿中的半开区间。
- `normalized_char_range`：句子在归一化稿件全文中的半开区间。
- `asr_token_range`：该句子匹配到的 ASR token index 范围。
- `diagnostics.matched_chars`：句子内成功匹配的归一化字符数。
- `diagnostics.total_normalized_chars`：句子归一化字符总数。
- `diagnostics.unmatched_manuscript_chars`：句子内未匹配的稿件字符。
- `diagnostics.candidate_count`：本句参与评分的候选窗口数量。
- `diagnostics.selected_candidate_rank`：最终候选在候选列表中的排名。
- `diagnostics.text_similarity`：句子归一化文本和候选 ASR 文本的 fuzzy 相似度。
- `diagnostics.candidate_window`：最终选中候选窗口覆盖的 ASR token 范围。
- `diagnostics.exact_matched_token_indexes`：候选窗口内与稿件字符完全相同的 ASR token，仅用于诊断，不直接决定最终时间范围。
- `diagnostics.timeline_provider`：生成该时间轴的 provider，例如 `hybrid`。
- `diagnostics.primary_timing_source`：主时间来源，hybrid 下为 `qwen3-forced`。
- `diagnostics.forced_unit_range`：forced alignment unit 范围。
- `diagnostics.asr_fuzzy`：hybrid 下保存 ASR fuzzy 句子摘要。

匹配阶段会做轻量数字读法兼容，例如稿件 `12元` 可与 ASR 的「十二元」对齐。该兼容只用于匹配和时间轴选择，不改变最终字幕文本。

当前示例要点：

- 第一句 `第一句话。` 时间范围为 `100` 到 `500` 毫秒。
- 第二句 `第二段有Ｅｎｇｌｉｓｈ 123！` 时间范围为 `600` 到 `2000` 毫秒。
- 两句 `status` 均为 `ok`，`match_score` 均为 `1.0`。

当前 ASR fuzzy 分支已使用顺序窗口 fuzzy 匹配，并要求最终句子时间范围不重叠。若 `raw_start_ms` 早于上一句最终 `end_ms`，会将当前句最终 `start_ms` 修正为上一句最终 `end_ms`，同时将 `time_adjusted` 标记为 `true`。

## `forced_alignment.json`

用途：保存 forced aligner 的标准化输出，便于复核模型返回文本与稿件归一化文本是否一致，以及每个 forced unit 的时间范围。

Schema：

```json
{
  "audio": {
    "path": "string",
    "format": "string",
    "duration_ms": "integer | null"
  },
  "aligner": {
    "provider": "string",
    "model": "string | null",
    "device_map": "string | null",
    "dtype": "string | null",
    "language": "string"
  },
  "input_text": "string",
  "normalized_text": "string",
  "forced_normalized_text": "string",
  "normalized_text_match": "boolean",
  "units": [
    {
      "index": "integer",
      "text": "string",
      "start_ms": "integer",
      "end_ms": "integer",
      "normalized_text": "string"
    }
  ],
  "diagnostics": {}
}
```

## `telemetry.json`

用途：保存 hybrid 置信度分析所需的双分支数据。第一版不做复杂自动仲裁，最终时间默认来自 forced alignment。

主要字段：

- `timeline_provider`：当前时间轴 provider。
- `primary`：主时间来源。
- `forced_alignment`：forced aligner 摘要，包含 provider、model、device、dtype、language、unit 数量和归一化文本一致性。
- `asr_fuzzy`：ASR fuzzy 摘要，包含 provider、model、token 数量和全文匹配分数。
- `sentences[].forced`：每句 forced 时间、unit 范围和状态。
- `sentences[].asr_fuzzy`：每句 ASR fuzzy 时间、token 范围、匹配文本和匹配分数。
- `sentences[].comparison`：forced 与 ASR fuzzy 的 start/end/duration 差值。

## `sentence_timeline.srt`

用途：保存由最终句子级时间轴渲染得到的 SRT 字幕文件。该产物通过 `render/srt.py` 中的 `SrtTimelineRenderer` 生成。

渲染会在不修改 `sentence_timeline.json` 的前提下填充阈值内的短间隙，并尽量延长过短字幕；具体修改记录在 `subtitle_render_report.json`。

格式示例：

```srt
1
00:00:00,410 --> 00:00:04,570
正是因为存在绝对正义，所以我们接受现实的相对正义。

2
00:00:05,230 --> 00:00:10,010
但是不要因为现实的相对正义，我们就认为这个世界没有正义。
```

渲染规则：

- 字幕序号从 `1` 开始，按 `sentence_timeline.json` 中的句子顺序递增。
- 时间范围使用最终 `start_ms` 和 `end_ms`，因此会继承无重叠修正后的结果。
- 时间格式为 SRT 标准 `HH:MM:SS,mmm`。
- 字幕正文使用稿件原句 `text`，不使用 ASR 文本。
- 缺失 `start_ms` 或 `end_ms` 的句子不会渲染为字幕块。

## `subtitle_render_report.json`

用途：保存 SRT 渲染阶段的闪轴修正诊断。`adjustments` 记录 `short_gap_filled` 和 `short_cue_extended`，`unresolved_short_cues` 记录因没有可用空闲时间而仍短于配置下限的字幕。

## `alignment_report.json`

用途：保存面向人工复核的汇总诊断报告。它聚合输入、ASR 摘要、归一化配置、句子切分配置、全局对齐结果和低置信度句子。

Schema：

```json
{
  "inputs": {
    "manuscript": "string",
    "audio": "string",
    "audio_format": "string"
  },
  "asr": {
    "provider": "string",
    "model": "string | null",
    "text": "string",
    "token_count": "integer"
  },
  "normalization": {
    "strategy": "string",
    "manuscript_normalized_text": "string",
    "asr_normalized_text": "string"
  },
  "segmentation": {
    "strategy": "string",
    "sentence_count": "integer",
    "sentences": ["SentenceSegment"]
  },
  "alignment": {
    "global_match_score": "number",
    "opcodes": ["AlignmentOpcode"],
    "unmatched_manuscript_chars": [
      {
        "normalized_index": "integer",
        "char": "string",
        "original_index": "integer",
        "original_char": "string"
      }
    ],
    "unmapped_asr_tokens": [
      {
        "normalized_index": "integer",
        "token_index": "integer",
        "token_text": "string",
        "normalized_char": "string",
        "start_ms": "integer",
        "end_ms": "integer"
      }
    ]
  },
  "low_confidence_sentences": ["SentenceTimelineItem"],
  "telemetry": {
    "timeline_provider": "string",
    "primary": "string",
    "forced_alignment": {},
    "asr_fuzzy": {}
  }
}
```

字段说明：

- `inputs`：本次运行的输入路径摘要。
- `asr`：ASR 服务、模型、文本和 token 数量摘要。
- `normalization.strategy`：归一化策略名称。
- `normalization.manuscript_normalized_text`：稿件归一化全文。
- `normalization.asr_normalized_text`：ASR 归一化全文。
- `segmentation`：句子切分策略和切分结果。
- `alignment.global_match_score`：全局匹配分数。
- `alignment.opcodes`：全局对齐操作。
- `alignment.unmatched_manuscript_chars`：稿件中未匹配的归一化字符，包含原稿位置。
- `alignment.unmapped_asr_tokens`：ASR 中未映射到稿件的字符或 token，包含时间范围。
- `low_confidence_sentences`：代码当前收集 `low_confidence`、`no_match`、`empty_after_normalization` 和 `invalid_time_range` 状态的完整句子时间轴条目。
- `telemetry`：可选 telemetry 摘要，包含双分支 provider、模型、数量和匹配分数，不嵌入完整 units 或 tokens。

当前示例要点：

- `alignment.unmapped_asr_tokens` 中包含 `嗯`，时间范围为 `0` 到 `100` 毫秒。
- `low_confidence_sentences` 为空。
- `segmentation.sentence_count` 为 `2`。

## 审阅建议

人工审阅时建议按以下顺序查看：

1. `sentence_timeline.json`：先看最终句子文本和时间范围是否符合预期。
2. `alignment_report.json`：查看低置信度句子、额外 ASR 内容和未匹配稿件字符。
3. `alignment.json`：需要追查错配时查看全局对齐操作。
4. `normalized_text.json`：需要确认归一化行为时查看原稿字符到归一化字符的映射。
5. `manuscript_segments.json`：需要调整句子切分规则时查看切分边界。
6. `word_timeline.json`：需要检查 ASR token 时间戳时查看原始时间轴。
