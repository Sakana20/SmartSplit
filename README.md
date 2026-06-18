# FunASR Timeline

本项目用于将 ASR 字符级或 token 级时间轴与 `.txt` 稿件对齐，并生成以稿件文本为准的句子级时间轴。

当前支持：

- `.txt` 分段纯文本稿件。
- `.mp3` 音频路径校验。
- mock ASR 时间轴输入。
- 本地 `paraformer-zh` FunASR 推理，默认模型目录为 `/Users/sakana/PyEnv/paraformer`。
- macOS MPS 推理，默认 `--paraformer-device mps`。
- 可通过 `--segmenter` 选择分句实现，当前内置 `regex` 和 `jieba-subtitle`。
- 支持 `[[NO_SPLIT]]...[[/NO_SPLIT]]` 标记保护不分句片段。
- 支持单独运行分句，导出一行一句的可编辑分句文本。
- 基础文本归一化。
- 顺序全局对齐。
- 顺序窗口 fuzzy 句子匹配。
- 匹配阶段支持轻量数字读法兼容，例如稿件 `12元` 可对齐 ASR 的「十二元」。
- 句子时间使用选中的 ASR 候选窗口完整 token 范围，替换字和数字读法差异不会造成时间轴缺口。
- SRT 字幕渲染。
- 丰富 JSON 输出和对齐诊断。

当前 ASR 实现：

- `mock`：读取 fixture `word_timeline.json`，用于常规测试和离线验证。
- `paraformer-zh`：使用 FunASR `AutoModel` 加载本地模型目录 `/Users/sakana/PyEnv/paraformer`，默认通过 macOS `mps` 推理。

当前输出文件：

- `word_timeline.json`
- `manuscript_segments.json`
- `normalized_text.json`
- `alignment.json`
- `sentence_timeline.json`
- `sentence_timeline.srt`
- `alignment_report.json`

完整使用方式见 [docs/usage.md](docs/usage.md)。每个输出文件的内容和 schema 见 [docs/output-artifacts.md](docs/output-artifacts.md)。

当前已固定一套真实 `paraformer-zh` 第一阶段结果样例，位于：

```text
tests/fixtures/stage1_paraformer/
```

这套 fixture 包含真实模型输入、当前 6 个 JSON 输出和 SRT 字幕输出，可作为第二阶段匹配逻辑的稳定样例。

## 第二阶段实现

第二阶段已基于 ground truth 稿件文本生成的音频这一前提，采用简单、顺序、可检查的方案：

- 分句可使用段落和强标点 `regex`，也可使用 `jieba-subtitle` 生成短视频字幕式短句。
- 每个分句从上一句结束 token 之后开始，在有限窗口内进行 fuzzy 匹配。
- fuzzy 评分先使用 Python 标准库 `difflib.SequenceMatcher`。
- 匹配后使用实际命中的首尾 token 合并句子时间范围。
- 最终 `sentence_timeline.json` 必须保证相邻句子的时间范围不重叠。
- 若时间被修正，应保留原始时间和 `time_adjusted` 诊断字段。

详细设计见 [docs/implementation-plan.md](docs/implementation-plan.md)。

## 常用命令

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

当前质量门禁状态：

- `pytest`：24 个测试通过。
- `ruff check`：通过。
- `ruff format --check`：通过。
- `mypy src`：通过。

## mock CLI 示例

```bash
uv run funasr-timeline \
  --manuscript tests/fixtures/manuscript.txt \
  --audio tests/fixtures/audio.mp3 \
  --output-dir /tmp/funasr-timeline-output \
  --segmenter jieba-subtitle \
  --asr-provider mock \
  --mock-word-timeline tests/fixtures/word_timeline.json
```

## 本地 Paraformer CLI 示例

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir /tmp/funasr-timeline-output \
  --segmenter jieba-subtitle \
  --asr-provider paraformer-zh \
  --paraformer-model-dir /Users/sakana/PyEnv/paraformer \
  --paraformer-device mps
```

## 单独分句与人工编辑

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --output-dir /tmp/funasr-segments \
  --segment-only \
  --segmenter jieba-subtitle
```

该命令会输出：

- `editable_segments.txt`：一行一句、空行分段的人类可读可编辑分句结果。
- `manuscript_segments.json`：带 offset 和归一化范围的结构化分句结果。

人工编辑后，可用 `--segments` 替代自动分句结果继续执行完整流程：

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --segments /tmp/funasr-segments/editable_segments.txt \
  --audio path/to/audio.mp3 \
  --output-dir /tmp/funasr-timeline-output \
  --asr-provider paraformer-zh
```

不分句保护标记：

```text
普通句子。[[NO_SPLIT]]这部分。不会被切开！会整体保留。[[/NO_SPLIT]]继续分句。
```
