# 使用说明

本文档记录当前项目的实际使用方式、输入要求、命令参数和输出产物。

## 基本能力

当前流程用于把 `.txt` ground truth 稿件和 `.mp3` 音频转换为以稿件文本为准的句子级时间轴，并输出 SRT 字幕。

主流程：

1. 读取 `.txt` 稿件。
2. 根据分句器生成稿件分句，或读取人工编辑后的分句结果。
3. 根据 `--timeline-provider` 运行 ASR fuzzy、Qwen3 forced alignment 或 hybrid 时间轴分支。
4. 对稿件文本、ASR 文本和 forced aligner 输出做基础归一化。
5. 将每个分句映射到 ASR token timeline 或 forced alignment unit timeline。
6. 合并每个分句的开始和结束时间，并保证相邻句子时间不重叠。
7. 输出 JSON 诊断产物和 `sentence_timeline.srt`。

当前实现重点面向“音频由 ground truth 稿件生成”的场景。CLI 默认使用 `hybrid` 时间轴策略：同时运行 Qwen3 forced alignment 和 ASR fuzzy 分支，最终时间默认采用 forced alignment，ASR fuzzy 结果作为 telemetry 保留。详细设计见 `docs/forced-aligner-plan.md`。

## 输入要求

### 稿件文本

- 格式：`.txt`
- 编码：UTF-8
- 内容：普通纯文本，可包含段落换行。
- 最终字幕正文默认以稿件文本为准。

示例：

```text
说实话，荔枝真的是夏天幸福感特别高的水果。
尤其最近正是季节，错过又得等一年。
```

### 音频

- 当前完整流程要求音频后缀为 `.mp3`。
- 真实 ASR 默认使用本地 `paraformer-zh` 模型。
- mock 流程只校验音频路径和 `.mp3` 后缀，不读取真实音频内容。

### 不分句保护标记

如果稿件中某段不希望被自动分句，可以用成对标记包住：

```text
普通句子。[[NO_SPLIT]]这部分。不会被切开！会整体保留。[[/NO_SPLIT]]继续分句。
```

规则：

- 开始标记：`[[NO_SPLIT]]`
- 结束标记：`[[/NO_SPLIT]]`
- 标记本身不会进入最终匹配文本或字幕。
- 标记内文本会作为一个完整分句输出。
- 开始和结束标记所在位置都是强制分句边界，标记前后的普通文本不会跨保护段合并。
- 使用 `llm` 分句时，保护段不会发送给模型；程序只发送两侧的普通文本，完成后再按原位置拼回保护段。
- 紧邻标记、位于保护段外侧的逗号或句号等边界标点不会单独成句，也不会并入保护段。
- 缺少结束标记、孤立结束标记或嵌套保护标记会明确报错。
- 保护标记应由上游直接写入输入 `.txt` 稿件，而不是在分句期间根据关键词动态添加。
- 如果同一份稿件还用于 TTS，合成语音前应调用 `remove_no_split_markers()` 得到去除标记但保留保护内容的朗读文本，避免把控制标记念出来。
- 对应 `manuscript_segments.json` 中的 `boundary` 为 `protected`。

例如：

```text
出门更省心。[[NO_SPLIT]]淘宝闪购最高12元无门槛红包可领取[[/NO_SPLIT]]，点击下方链接了解更多。
```

保护段会固定独占一句；LLM 分别处理“出门更省心”和“点击下方链接了解更多。”，结束标记后的逗号只作为接缝边界处理。

## 分句实现

当前通过 `--segmenter` 选择分句实现。

### `hanlp`

`hanlp` 使用 HanLP 中文多任务模型执行 constituency 解析，并从 constituency tree
的叶节点取得 token。程序先按 `，,、。！？!?；;：:` phrase 边界拆句，再按 token 顺序拼接；加入
下一个 token 会让折算长度超过 `--segment-threshold`（默认 10）时开始新句。汉字计 1，
英文和数字每两个计 1，标点、空白、符号及分隔符不计；单个超长 token 不会被拆开。
模型由 HanLP 下载并缓存，后续运行复用本地缓存。

### `regex`

按段落和强标点切分。

强边界：

- `。`
- `！`
- `？`
- `!`
- `?`
- `；`
- `;`
- 段落换行

适合保留自然句，不按逗号切分。

### `jieba-subtitle`

面向短视频字幕的短句分割。

规则：

- 先按段落和强标点形成基础范围。
- 逗号、顿号、句号、问号、感叹号、分号等标点会作为短语边界，但不会进入最终分句文本。
- 再使用 jieba 分词拼接短句。
- 默认目标为单句归一化文本不超过 10 个字符。
- 同一个 jieba 词不会被拆到两个分句中。
- 如果某个词本身超过目标长度，则保留完整词，允许该分句超过 10 个字符。
- 保护标记包住的内容不会进入 jieba 分句。
- 内置了少量短视频字幕常见软切分点，例如 `特别`、`直接`，用于把过长表达拆成更自然的语义短语。

示例：

```text
说实话，荔枝真的是夏天幸福感特别高的水果
```

可能被切为：

```text
说实话
荔枝真的是夏天幸福感
特别高的水果
```

如果希望某段超过 10 个字仍整体保留，应使用保护标记：

```text
[[NO_SPLIT]]现在淘宝闪购有最高12元无门槛红包[[/NO_SPLIT]]
```

## 完整流程命令

### Hybrid Forced Aligner 模式

默认完整流程会读取 `configs/aligner-qwen3.toml`。

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir path/to/output \
  --segmenter jieba-subtitle \
  --timeline-provider hybrid \
  --aligner-config configs/aligner-qwen3.toml
```

aligner 配置文件包含 Qwen3 forced aligner 和 ASR fuzzy 两部分配置。默认 Qwen3 配置为本地模型 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`、`device_map = "mps"`、`dtype = "bfloat16"`、`language = "Chinese"`。

### 使用本地 Paraformer

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir path/to/output \
  --segmenter jieba-subtitle \
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh \
  --paraformer-model-dir /Users/sakana/PyEnv/paraformer \
  --paraformer-device mps
```

如果只想运行旧 ASR fuzzy 流程，可显式指定：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir path/to/output \
  --segmenter jieba-subtitle \
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh
```

### 使用 mock ASR fixture

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

mock 流程适合测试分句、匹配、合并、渲染和输出 schema，不需要真实 ASR 推理。

## 单独分句

如果只想先看自动分句效果，可以运行：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --output-dir path/to/segments \
  --segment-only \
  --segmenter jieba-subtitle
```

输出：

- `editable_segments.txt`
- `manuscript_segments.json`

`editable_segments.txt` 是一行一句、空行分段的文本文件，可人工编辑。

示例：

```text
说实话
荔枝真的是夏天幸福感
特别高的水果

尤其最近
正是季节
```

## 使用编辑后的分句结果继续流程

编辑 `editable_segments.txt` 后，可以通过 `--segments` 让完整流程跳过自动分句：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --segments path/to/editable_segments.txt \
  --audio path/to/audio.mp3 \
  --output-dir path/to/output \
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh \
  --paraformer-model-dir /Users/sakana/PyEnv/paraformer \
  --paraformer-device mps
```

注意：

- `--segments` 文件内容会成为后续匹配和字幕输出使用的分句 ground truth。
- `--manuscript` 仍然需要传入，用于记录输入路径和保持命令结构一致。
- 编辑分句时建议只调整换行位置；如果修改文字，应确保和音频内容一致。

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--manuscript` | 是 | 无 | `.txt` 稿件路径。 |
| `--audio` | 完整流程必填 | 无 | `.mp3` 音频路径。`--segment-only` 时不需要。 |
| `--subtitle-alignment-audio` | 否 | `--audio` | 渲染 SRT 时用于将最后一条有效字幕的结束时间对齐到音频结尾。TTS 音频经过格式转换时，应传入转换前的原始音频。 |
| `--no-align-last-subtitle-to-audio-end` | 否 | `false` | 关闭末条字幕结束时间到音频结尾的对齐。 |
| `--no-align-first-subtitle-to-audio-start` | 否 | `false` | 关闭首条有效字幕开始时间到音频起点 `00:00:00,000` 的对齐。 |
| `--subtitle-gap-threshold-ms` | 否 | `67` | 空白闪轴阈值；相邻字幕的正间隙不超过该值时延长上一条字幕以填满间隙，`0` 表示关闭。 |
| `--subtitle-min-duration-ms` | 否 | `200` | 渲染字幕最短持续时间；短字幕在不重叠的前提下利用相邻空闲时间延长，`0` 表示关闭。 |
| `--output-dir` | 是 | 无 | 输出目录。不存在会自动创建。 |
| `--segmenter` | 否 | `regex` | 分句实现，可选 `regex`、`hanlp`、`jieba-subtitle`、`llm`。 |
| `--segment-threshold` | 否 | `10` | `hanlp` 分句的有效字符数阈值。 |
| `--llm-config` | `--segmenter llm` 时读取 | `configs/llm-siliconflow.toml` | OpenAI-compatible LLM 分句配置文件。 |
| `--llm-fallback-segmenter` | 否 | `hanlp` | LLM block 重试耗尽后的 fallback，可选 `hanlp`、`jieba-subtitle`、`regex`。 |
| `--llm-raise-on-error` | 否 | `false` | 重试耗尽后直接抛错并停止，不执行 block fallback。 |
| `--segment-only` | 否 | `false` | 只运行分句，输出可编辑分句文本和结构化分句 JSON。 |
| `--segments` | 否 | 无 | 使用人工编辑后的一行一句文本替代自动分句结果。 |
| `--timeline-provider` | 否 | 配置文件中 `timeline.provider`，未配置为 `hybrid` | 时间轴来源，可选 `asr-fuzzy`、`qwen3-forced`、`hybrid`。 |
| `--aligner-config` | 否 | `configs/aligner-qwen3.toml` | forced aligner 与 hybrid 配置文件。 |
| `--asr-provider` | 否 | 配置文件中 `asr.provider`，未配置为 `paraformer-zh` | ASR 服务，可选 `mock`、`paraformer-zh`。 |
| `--mock-word-timeline` | mock 必填 | 无 | mock ASR 使用的 `word_timeline.json` 路径。 |
| `--paraformer-model-dir` | 否 | `/Users/sakana/PyEnv/paraformer` | 本地 `paraformer-zh` 模型目录。 |
| `--paraformer-device` | 否 | `mps` | 推理设备，例如 `mps`、`cpu`、`cuda:0`。 |
| `--quiet` | 否 | `false` | 关闭默认 debug 日志，仅保留命令行错误输出。 |

CLI 默认启用 debug 日志，便于检查稿件读取、ASR、分句、归一化、匹配、合并和文件写入等阶段。命令结束时会用 Rich 表格展示输出路径；该表格只影响终端显示，不会改变 JSON、SRT 或其他产物内容。

如果 ASR 使用由 TTS 原始音频转换得到的 MP3，建议显式传入原始音频：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/converted.mp3 \
  --subtitle-alignment-audio path/to/original.ogg \
  --output-dir path/to/output
```

首条字幕默认从 `00:00:00,000` 开始，末条字幕默认对齐音频结尾并向上取整到 30fps 帧边界。两项对齐都只在最终 SRT 渲染时生效；`sentence_timeline.json`、匹配结果、诊断字段和主流程时间轴均保留原值。

## LLM 分句配置

`llm` 分句使用 OpenAI-compatible Chat Completions 协议。服务商只通过 `base_url` 配置，不和具体厂商协议耦合。

配置文件见 `configs/llm-siliconflow.toml`：

```toml
[llm]
base_url = "https://api.siliconflow.cn/v1"
model = "Qwen/Qwen3.5-4B"
api_key_env = "FUNASR_TIMELINE_LLM_API_KEY"
timeout_seconds = 240
temperature = 0
max_tokens = 8192
enable_thinking = false
```

如果需要切换模型，只修改配置文件即可，CLI 仍通过 `--llm-config` 读取。

本地使用时通过环境变量提供密钥：

```bash
export FUNASR_TIMELINE_LLM_API_KEY="sk-..."
uv run funasr-timeline \
  --manuscript path/to/input.txt \
  --output-dir path/to/segments \
  --segment-only \
  --segmenter llm \
  --llm-config configs/llm-siliconflow.toml
```

程序会为每个预处理后的 block 创建一个独立 LLM 请求，并以 block 数量作为并发数同时发送；每个请求只包含一个 block，因此不依赖模型复述 block id 或输出跨 block 分隔符。请求默认设置 `enable_thinking = false`，程序只解析最终 `message.content`，不读取或回退到推理字段。调用模型前，程序会先移除保护段并把标记两侧设为强制 block 边界；紧邻标记的外侧边界标点从 LLM 视图中裁掉，但原稿 offset 保持不变。

每个 block 独立执行内容覆盖和长度校验。覆盖校验会分别删除原文与模型结果中的标点、空格、换行和其他分隔符，并对剩余字母、中文和数字进行 Unicode 归一化后比较。因此模型在 `iPhone15`、`500ml` 或数字单位周围新增空格可以通过校验，但不能漏字、改字或改变数字。校验通过后，程序不会直接采用模型文本，而是根据内容字符顺序定位回原稿 offset，从原稿切片生成最终分句，再去掉分句两端的边界标点并插回保护段；模型新增的空格和标点不会进入最终字幕。

prompt 会强调常规口播字幕的折算长度优先控制在 4 到 8 个汉字，硬上限为 10；汉字计 1，英文和数字每两个计 1，标点符号不计。请求异常、响应解析错误和内容校验错误均按 block 独立重试；通过校验的 block 不参与后续反馈。重试耗尽后默认记录 error 日志，并只把失败 block 交给 `--llm-fallback-segmenter`（默认 `hanlp`），成功 block 仍采用 LLM 结果。`--llm-raise-on-error` 可关闭 fallback 并直接抛错。完整流程和独立分句会在输出目录写出 `llm_segmentation_diagnostics.json`，包含每个 block 的请求、重试、失败原因和最终策略。`manuscript_segments.json` 的 `segmenter` 与 `source_block_id` 可用于后续 telemetry。`[[NO_SPLIT]]...[[/NO_SPLIT]]` 保护段由代码强制整体保留，不依赖模型遵守。

## 匹配和时间轴规则

`asr-fuzzy` 流程会把分句后的稿件文本顺序匹配到 ASR token timeline。`qwen3-forced` 流程会一次性对齐分句后的完整稿件文本，再按归一化 offset 把 forced alignment units 映射回分句。`hybrid` 会同时运行两者，并以 forced alignment 时间作为最终主时间。

当前规则：

- 以稿件分句文本作为最终字幕文本。
- 每个分句从上一句结束 token 之后继续搜索，保证整体顺序单调。
- 使用 `difflib.SequenceMatcher` 对候选 ASR 窗口计算 fuzzy 相似度。
- 最终时间使用选中的 ASR 候选窗口完整 token 范围，而不是只使用完全相同字符的 token。
- 字符级完全匹配 token 会写入 `diagnostics.exact_matched_token_indexes`，用于诊断。
- 相邻分句最终时间范围会做无重叠修正。
- 匹配阶段支持轻量数字读法兼容，例如稿件 `12元` 可以对齐 ASR 的「十二元」。
- 数字读法兼容只用于匹配和时间选择，不改变字幕正文。
- forced aligner 输出如果跳过标点，会通过项目归一化 offset 继续映射到稿件分句。
- hybrid 模式会把 ASR fuzzy 的句子时间、匹配分数和 token 范围写入 telemetry 和 `diagnostics.asr_fuzzy`。

常见状态：

| 状态 | 含义 |
| --- | --- |
| `ok` | 匹配正常。 |
| `low_confidence` | 匹配成功但相似度低，需要人工复核。 |
| `no_match` | 没有找到可用候选。 |
| `empty_after_normalization` | 分句归一化后为空，例如只有标点。 |
| `invalid_time_range` | 时间范围非法，通常需要检查 ASR token 或匹配结果。 |
| `forced_missing_unit` | forced alignment 输出无法覆盖该分句归一化范围。 |
| `forced_empty_segment` | 分句归一化后为空，无法映射 forced units。 |
| `forced_invalid_time_range` | forced alignment 单元产生非法时间范围。 |

## 输出文件

完整流程会生成：

```text
word_timeline.json
manuscript_segments.json
normalized_text.json
alignment.json
sentence_timeline.json
sentence_timeline.srt
alignment_report.json
```

`qwen3-forced` 或 `hybrid` 模式还会生成：

```text
forced_alignment.json
telemetry.json
```

单独分句流程会生成：

```text
editable_segments.txt
manuscript_segments.json
```

### `word_timeline.json`

ASR token 级时间轴。

主要内容：

- 音频路径、格式和时长。
- ASR provider、model 和完整识别文本。
- token 列表，每个 token 包含 `index`、`text`、`start_ms`、`end_ms`、`confidence`、`source`。

### `manuscript_segments.json`

稿件分句结果。

主要字段：

- `index`：分句序号。
- `text`：分句原文。
- `paragraph_index`：段落序号。
- `char_start` / `char_end`：在当前用于匹配的稿件文本中的半开区间。
- `boundary`：边界来源，常见值为 `punctuation`、`paragraph`、`protected`、`editable`。
- `segmenter`：实际生成该句的分句器；LLM 失败 block 会显示 fallback 分句器。
- `source_block_id`：LLM 编排对应的 `block-N`，非 LLM 流程为 `null`。
- `normalized_text`：用于匹配的归一化文本。
- `normalized_start` / `normalized_end`：在归一化稿件全文中的半开区间。

### `normalized_text.json`

稿件和 ASR 的归一化文本与 offset 映射。

用于检查：

- 标点、空白和全半角处理。
- 原稿字符到归一化字符的映射。
- ASR token 到归一化字符的映射。

### `alignment.json`

全文级顺序对齐结果。

主要内容：

- `global_match_score`
- `manuscript_to_token`
- `opcodes`
- `unmatched_manuscript_indexes`
- `unmapped_asr_indexes`

### `sentence_timeline.json`

最终句子级时间轴，是最主要的业务 JSON。

主要字段：

- `text`：最终字幕文本，来自稿件分句。
- `start_ms` / `end_ms`：最终无重叠时间范围。
- `raw_start_ms` / `raw_end_ms`：匹配直接得到的原始时间范围。
- `time_adjusted`：是否为避免重叠而调整过时间。
- `match_score`：句子与候选 ASR 文本的 fuzzy 相似度。
- `status`：常见值为 `ok`、`low_confidence`、`no_match`、`empty_after_normalization`、`invalid_time_range`。
- `matched_token_indexes`：用于取时间的 ASR 候选窗口 token index，包含替换字和数字读法差异对应的 ASR token。
- `matched_asr_text`：被选中的 ASR 候选窗口文本。
- `diagnostics`：候选数量、相似度、未匹配字符等诊断信息。
- hybrid 模式下 `start_ms` / `end_ms` 默认来自 forced alignment；`diagnostics.primary_timing_source` 为 `qwen3-forced`，`diagnostics.asr_fuzzy` 保存 ASR fuzzy 摘要。

### `forced_alignment.json`

Qwen3 forced alignment 标准化结果。

主要内容：

- aligner provider、model、device、dtype 和 language。
- 输入文本、稿件归一化文本和 aligner 输出归一化文本。
- `normalized_text_match`：两者是否完全一致。
- `units`：每个 forced alignment unit 的 `index`、`text`、`normalized_text`、`start_ms`、`end_ms`。

### `telemetry.json`

hybrid 分析数据。

主要内容：

- `timeline_provider` 和 `primary`。
- forced alignment 摘要和可选 units。
- ASR fuzzy 摘要和可选 tokens。
- 每句 forced 时间、ASR fuzzy 时间和两者差值。

匹配阶段会做轻量数字读法归一化，例如 `12元` 可与 ASR 的 `十二元` 对齐。该归一化只用于匹配和时间轴选择，不改变最终字幕文本。

### `sentence_timeline.srt`

由 `sentence_timeline.json` 渲染得到的字幕文件。

规则：

- 序号从 `1` 开始。
- 时间格式为 `HH:MM:SS,mmm`。
- 字幕正文使用 `sentence_timeline.json` 中的 `text`。
- 缺失时间或非法时间范围的句子不会渲染。
- 默认填充不超过 67ms 的相邻字幕空隙，避免字幕短暂消失形成空白闪轴。
- 默认把短于 200ms 的字幕向相邻空闲时间延长；不会移动其他字幕、合并文本或制造重叠。
- 闪轴处理在首尾音频对齐后执行，并且只改变 SRT 渲染副本，不回写 `sentence_timeline.json`。

### `subtitle_render_report.json`

记录渲染后处理配置、空白闪轴填充、短字幕延长，以及因相邻时间空间不足而未达到最短时长的 cue。`source_index` 对应 `sentence_timeline.json` 中的句子 `index`。

### `alignment_report.json`

人工复核用汇总报告。

主要内容：

- 输入路径摘要。
- ASR 摘要。
- 归一化摘要。
- 分句策略和分句结果。
- 全局对齐摘要。
- 低置信度或无时间句子。
- 未匹配稿件字符和未映射 ASR token。

## 推荐审阅顺序

1. `sentence_timeline.srt`：先看字幕是否符合预期。
2. `sentence_timeline.json`：检查时间、匹配分数和 `time_adjusted`。
3. `alignment_report.json`：查看低置信度、未匹配字符和 ASR 差异。
4. `manuscript_segments.json`：检查分句边界。
5. `normalized_text.json`：检查归一化和 offset。
6. `word_timeline.json`：检查 ASR token 时间戳。

## 质量命令

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

当前项目要求这些命令通过后再认为实现完成。

## 端到端测试

`tests/e2e/test_jianying_smartsplit_demo.py` 默认运行真实 demo 链路：复用剪映 demo 中的长中文混合文本，调用剪映 TTS 生成音频，使用 LLM 分句，通过 `hybrid` 同时运行 Qwen3 forced aligner 和本地 `paraformer-zh`/FunASR ASR，并可把音频和 SRT 写回剪映草稿。测试成功时会额外写出 `e2e_diagnostics.json`，记录命令、文本长度、TTS 信息、音频转换、剪映草稿、句子数量、状态分布、telemetry 摘要和 report 摘要；子命令失败时会写出 `e2e_failure_diagnostics.json`，保留 TTS、音频转换和 stdout/stderr 尾部输出，方便定位。

运行前需要本机具备可 import 的剪映 Python 接口、TTS 后端、本地 `paraformer-zh`、本地 Qwen3 forced aligner、可用 LLM API key 和配置：

```bash
set -a
source configs/jianying-e2e.env
set +a

uv run pytest tests/e2e/test_jianying_smartsplit_demo.py -q
```

需要观察长链路实时日志时使用 `-s`：

```bash
uv run pytest tests/e2e/test_jianying_smartsplit_demo.py -s
```

测试中的 ffmpeg 与 `funasr_timeline.cli` 子进程会把 stdout/stderr 实时透传到当前
pytest 终端，同时继续把完整逐行输出保存在 `e2e_diagnostics.json` 或失败诊断中。

真实测试环境变量：

| 环境变量 | 默认/示例 | 说明 |
| --- | --- | --- |
| `FUNASR_TIMELINE_E2E_JIANYING_SCRIPTS_PATH` | 无 | 可选。若 `jy_wrapper` 和 `universal_tts` 不能被当前 Python 环境直接 import，则填入包含这两个模块的脚本目录。 |
| `FUNASR_TIMELINE_E2E_DRAFT` | `SmartSplit_E2E_Test` | 测试草稿名；测试会重建同名草稿。 |
| `FUNASR_TIMELINE_E2E_VOICE_ID` | `BV005_streaming` | TTS 音色 ID。 |
| `FUNASR_TIMELINE_E2E_LLM_CONFIG` | `configs/llm-siliconflow.toml` | LLM 分句配置。 |
| `FUNASR_TIMELINE_LLM_API_KEY` | 无 | `configs/llm-siliconflow.toml` 默认读取的 API key 环境变量。 |
| `FUNASR_TIMELINE_E2E_ALIGNER_CONFIG` | `configs/aligner-qwen3.toml` | 真实 forced aligner 与 ASR 配置。 |
| `FUNASR_TIMELINE_E2E_WRITE_DRAFT` | `1` | 设置为 `0` 时只生成 TTS 和时间轴，不把音频/SRT 写回剪映草稿。 |

配置文件处理建议：

- `configs/llm-siliconflow.toml` 直接作为默认 LLM 配置使用，密钥仍通过 `FUNASR_TIMELINE_LLM_API_KEY` 提供。
- `configs/aligner-qwen3.toml` 的模型路径已经按当前本机约定填写为 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B` 和 `/Users/sakana/PyEnv/paraformer`。真实测试前请确认路径、`device_map = "mps"`、`dtype = "bfloat16"` 与当前环境一致。
- `configs/jianying-e2e.env` 保存当前本机 demo e2e 的可直接运行环境变量，包括 LLM key 环境变量赋值。该文件适合本地验证；共享仓库或提交到公开环境前应移除或替换敏感值。

当前状态：

```text
uv run pytest
# 默认会运行真实 demo e2e，需要本地模型、TTS、剪映 Python 接口和 LLM API key。
```

只运行常规确定性测试时可排除真实端到端测试：

```bash
uv run pytest -m 'not e2e_real'
```
