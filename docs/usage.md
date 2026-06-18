# 使用说明

本文档记录当前项目的实际使用方式、输入要求、命令参数和输出产物。

## 基本能力

当前流程用于把 `.txt` ground truth 稿件和 `.mp3` 音频转换为以稿件文本为准的句子级时间轴，并输出 SRT 字幕。

主流程：

1. 读取 `.txt` 稿件。
2. 根据分句器生成稿件分句，或读取人工编辑后的分句结果。
3. 通过 ASR 服务生成 token 级时间轴。
4. 对稿件文本和 ASR 文本做基础归一化。
5. 将每个分句顺序 fuzzy 匹配到 ASR token timeline。
6. 使用选中的 ASR 候选窗口完整 token 范围合并每个分句的开始和结束时间，并保证相邻句子时间不重叠。
7. 输出 JSON 诊断产物和 `sentence_timeline.srt`。

当前实现重点面向“音频由 ground truth 稿件生成”的场景，因此默认按稿件顺序做局部 fuzzy 匹配，不按多人对话、插话或大段错序音频设计。

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
- 对应 `manuscript_segments.json` 中的 `boundary` 为 `protected`。

## 分句实现

当前通过 `--segmenter` 选择分句实现。

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

### 使用本地 Paraformer

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir path/to/output \
  --segmenter jieba-subtitle \
  --asr-provider paraformer-zh \
  --paraformer-model-dir /Users/sakana/PyEnv/paraformer \
  --paraformer-device mps
```

### 使用 mock ASR fixture

```bash
uv run funasr-timeline \
  --manuscript tests/fixtures/manuscript.txt \
  --audio tests/fixtures/audio.mp3 \
  --output-dir test_temp \
  --segmenter regex \
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
| `--output-dir` | 是 | 无 | 输出目录。不存在会自动创建。 |
| `--segmenter` | 否 | `regex` | 分句实现，可选 `regex`、`jieba-subtitle`。 |
| `--segment-only` | 否 | `false` | 只运行分句，输出可编辑分句文本和结构化分句 JSON。 |
| `--segments` | 否 | 无 | 使用人工编辑后的一行一句文本替代自动分句结果。 |
| `--asr-provider` | 否 | `mock` | ASR 服务，可选 `mock`、`paraformer-zh`。 |
| `--mock-word-timeline` | mock 必填 | 无 | mock ASR 使用的 `word_timeline.json` 路径。 |
| `--paraformer-model-dir` | 否 | `/Users/sakana/PyEnv/paraformer` | 本地 `paraformer-zh` 模型目录。 |
| `--paraformer-device` | 否 | `mps` | 推理设备，例如 `mps`、`cpu`、`cuda:0`。 |

## 匹配和时间轴规则

完整流程会把分句后的稿件文本顺序匹配到 ASR token timeline。

当前规则：

- 以稿件分句文本作为最终字幕文本。
- 每个分句从上一句结束 token 之后继续搜索，保证整体顺序单调。
- 使用 `difflib.SequenceMatcher` 对候选 ASR 窗口计算 fuzzy 相似度。
- 最终时间使用选中的 ASR 候选窗口完整 token 范围，而不是只使用完全相同字符的 token。
- 字符级完全匹配 token 会写入 `diagnostics.exact_matched_token_indexes`，用于诊断。
- 相邻分句最终时间范围会做无重叠修正。
- 匹配阶段支持轻量数字读法兼容，例如稿件 `12元` 可以对齐 ASR 的「十二元」。
- 数字读法兼容只用于匹配和时间选择，不改变字幕正文。

常见状态：

| 状态 | 含义 |
| --- | --- |
| `ok` | 匹配正常。 |
| `low_confidence` | 匹配成功但相似度低，需要人工复核。 |
| `no_match` | 没有找到可用候选。 |
| `empty_after_normalization` | 分句归一化后为空，例如只有标点。 |
| `invalid_time_range` | 时间范围非法，通常需要检查 ASR token 或匹配结果。 |

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

匹配阶段会做轻量数字读法归一化，例如 `12元` 可与 ASR 的 `十二元` 对齐。该归一化只用于匹配和时间轴选择，不改变最终字幕文本。

### `sentence_timeline.srt`

由 `sentence_timeline.json` 渲染得到的字幕文件。

规则：

- 序号从 `1` 开始。
- 时间格式为 `HH:MM:SS,mmm`。
- 字幕正文使用 `sentence_timeline.json` 中的 `text`。
- 缺失时间或非法时间范围的句子不会渲染。

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

当前状态：

```text
uv run pytest
# 24 passed, 1 warning
```
