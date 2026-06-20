# FunASR Timeline

本项目用于将 ASR 字符级或 token 级时间轴与 `.txt` 稿件对齐，并生成以稿件文本为准的句子级时间轴。

当前支持：

- `.txt` 分段纯文本稿件。
- `.mp3` 音频路径校验。
- mock ASR 时间轴输入。
- 本地 `paraformer-zh` FunASR 推理，默认模型目录为 `/Users/sakana/PyEnv/paraformer`。
- macOS MPS 推理，默认 `--paraformer-device mps`。
- 可通过 `--segmenter` 选择分句实现，当前内置 `regex`、`jieba-subtitle` 和可选在线 `llm`。
- LLM 分句采用纯文本换行输出，不使用 XML/JSON；输出必须完整保留原文标点，校验通过后再由程序去掉分句两端边界标点。
- LLM 分句默认要求每句 4 到 12 个中文字符左右，硬上限为 14 个中文字符；英文、数字和标点不计入长度，超长会触发反馈重试。
- 支持 `[[NO_SPLIT]]...[[/NO_SPLIT]]` 标记保护不分句片段。
- 支持单独运行分句，导出一行一句的可编辑分句文本。
- 基础文本归一化。
- 顺序全局对齐。
- 顺序窗口 fuzzy 句子匹配。
- 匹配阶段支持轻量数字读法兼容，例如稿件 `12元` 可对齐 ASR 的「十二元」。
- 句子时间使用选中的 ASR 候选窗口完整 token 范围，替换字和数字读法差异不会造成时间轴缺口。
- 本地 Qwen3 forced aligner，默认模型目录为 `/Users/sakana/PyEnv/Qwen3-ForcedAligner-0.6B`。
- `asr-fuzzy`、`qwen3-forced` 和 `hybrid` 三种时间轴策略。默认配置为 `hybrid`，最终时间以 Qwen3 forced alignment 为主，ASR fuzzy 进入 telemetry。
- SRT 字幕渲染。
- 丰富 JSON 输出和对齐诊断。

当前 ASR 实现：

- `mock`：读取 fixture `word_timeline.json`，用于常规测试和离线验证。
- `paraformer-zh`：使用 FunASR `AutoModel` 加载本地模型目录 `/Users/sakana/PyEnv/paraformer`，默认通过 macOS `mps` 推理。

当前 forced alignment 实现：

- `mock`：读取 fixture forced units，供常规测试和 CLI hybrid 测试使用。
- `qwen3-forced`：使用 `qwen_asr.Qwen3ForcedAligner` 加载本地 `Qwen3-ForcedAligner-0.6B`，默认 `device_map = "mps"`、`dtype = "bfloat16"`、`language = "Chinese"`。

当前输出文件：

- `word_timeline.json`
- `manuscript_segments.json`
- `normalized_text.json`
- `alignment.json`
- `sentence_timeline.json`
- `sentence_timeline.srt`
- `alignment_report.json`

`qwen3-forced` 或 `hybrid` 模式还会输出：

- `forced_alignment.json`
- `telemetry.json`

完整使用方式见 [docs/usage.md](docs/usage.md)。每个输出文件的内容和 schema 见 [docs/output-artifacts.md](docs/output-artifacts.md)。

当前已固定一套真实 `paraformer-zh` ASR fuzzy 结果样例，位于：

```text
tests/fixtures/stage1_paraformer/
```

这套 fixture 包含真实模型输入、ASR fuzzy 输出 JSON 和 SRT 字幕输出，可作为 ASR fuzzy 匹配逻辑的稳定样例。

## 当前时间轴流程

当前流程基于 ground truth 稿件文本生成音频这一前提，采用简单、顺序、可检查的方案：

- 分句可使用段落和强标点 `regex`，也可使用 `jieba-subtitle` 生成短视频字幕式短句。
- 需要短视频口播风格时，可使用 `llm` 分句并通过 `configs/llm-siliconflow.toml` 配置 OpenAI-compatible 接口。
- 每个分句从上一句结束 token 之后开始，在有限窗口内进行 fuzzy 匹配。
- fuzzy 评分先使用 Python 标准库 `difflib.SequenceMatcher`。
- 匹配后使用实际命中的首尾 token 合并句子时间范围。
- `qwen3-forced` 会一次性对齐分句后的完整稿件文本，再按归一化 offset 回填到每个分句。
- `hybrid` 会同时保留 forced 和 ASR fuzzy 两条分支；最终时间默认来自 forced alignment。
- 最终 `sentence_timeline.json` 必须保证相邻句子的时间范围不重叠。
- 若时间被修正，应保留原始时间和 `time_adjusted` 诊断字段。

详细设计见 [docs/implementation-plan.md](docs/implementation-plan.md)。

## 常用命令

```bash
uv sync
uv run pytest -m 'not e2e_real'
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

完整真实端到端链路可单独运行：

```bash
uv run pytest tests/e2e/test_jianying_smartsplit_demo.py -q
```

当前测试收集情况和质量命令：

- `pytest --collect-only`：当前收集 40 个测试，其中包含 1 个标记为 `e2e_real` 的真实剪映/LLM/Qwen3/FunASR 端到端测试。
- 常规确定性测试可使用 `uv run pytest -m 'not e2e_real'`。
- 完整真实链路测试需要本地模型、TTS、剪映 Python 接口和 LLM API key。
- 代码变更时仍应运行 `ruff check`、`ruff format --check` 和 `mypy src`。

## mock CLI 示例

```bash
uv run funasr-timeline \
  --manuscript tests/fixtures/manuscript.txt \
  --audio tests/fixtures/audio.mp3 \
  --output-dir /tmp/funasr-timeline-output \
  --segmenter jieba-subtitle \
  --timeline-provider asr-fuzzy \
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
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh \
  --paraformer-model-dir /Users/sakana/PyEnv/paraformer \
  --paraformer-device mps
```

## Hybrid Qwen3 + Paraformer CLI 示例

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --audio path/to/audio.mp3 \
  --output-dir /tmp/funasr-timeline-output \
  --segmenter llm \
  --llm-config configs/llm-siliconflow.toml \
  --timeline-provider hybrid \
  --aligner-config configs/aligner-qwen3.toml
```

## 单独分句与人工编辑

```bash
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --output-dir /tmp/funasr-segments \
  --segment-only \
  --segmenter jieba-subtitle
```

LLM 分句使用 OpenAI-compatible Chat Completions 端点，通过 TOML 配置读取：

```bash
export FUNASR_TIMELINE_LLM_API_KEY="sk-..."
uv run funasr-timeline \
  --manuscript path/to/manuscript.txt \
  --output-dir /tmp/funasr-segments \
  --segment-only \
  --segmenter llm \
  --llm-config configs/llm-siliconflow.toml
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
  --timeline-provider asr-fuzzy \
  --asr-provider paraformer-zh
```

不分句保护标记：

```text
普通句子。[[NO_SPLIT]]这部分。不会被切开！会整体保留。[[/NO_SPLIT]]继续分句。
```
