# 研究摘要：FunASR 时间轴对齐

## 目标

本项目的目标是构建一条以稿件为准的音频时间轴处理流程：

1. 使用实现了统一接口的 ASR 服务，从配音音频中生成字符级或 token 级时间戳。
2. 将 ASR 输出文本与已有 `.txt` 稿件进行顺序对齐。
3. 将字符级时间范围合并为句子级时间范围。
4. 输出丰富的 JSON 结果和诊断报告，便于人工复核和后续阶段优化。

当前先聚焦 `.txt` 稿件和统一 MP3 音频，并已接入本地 `paraformer-zh` ASR 服务和本地 Qwen3 forced aligner。非 MP3 音频由 ffmpeg 自动转换为 MP3；`paraformer-zh` 内部使用 FunASR `AutoModel`；Qwen3 forced aligner 通过 `qwen_asr.Qwen3ForcedAligner` 接入。

## FunASR 时间戳能力

中文配音场景推荐优先研究 FunASR 的以下模型组合：

- ASR 模型：`paraformer-zh`
- VAD 模型：`fsmn-vad`
- 标点模型：`ct-punc`

FunASR 的 Paraformer 示例通常会在结果对象中暴露 timestamp 信息，常见形式类似 `res[0]["timestamp"]`。中文语音场景下，该信息可用于构造字符级或 token 级时间轴，时间单位通常为毫秒，例如：

```json
[[880, 1120], [1120, 1320]]
```

因此 `paraformer-zh` 适合作为第一类真实 ASR 服务实现：输入音频，输出识别文本和每个字符或 token 的时间范围。项目层面不把服务命名为 `funasr`，因为 FunASR 是模型集合与推理框架，不是单个模型实现。

当前项目使用的本地模型目录为：

```text
/Users/sakana/PyEnv/paraformer
```

该目录包含 `config.yaml`、`model.pt`、`tokens.json`、`am.mvn` 等 FunASR 本地加载所需文件。实测 `AutoModel(model="/Users/sakana/PyEnv/paraformer", device="mps")` 可在 macOS MPS 上运行，并返回包含 `text` 和 `timestamp` 的结果。项目实现文件为 `src/funasr_timeline/asr/paraformer_zh_service.py`。

参考：

- https://github.com/modelscope/FunASR
- https://modelscope.github.io/FunASR/tutorial.html

## 统一 ASR 接口

项目不应让主流程直接依赖某一个 ASR SDK 的返回结构。应定义统一接口，由具体服务实现：

- `paraformer-zh` 服务实现。
- mock 服务实现，用于测试和 fixture 驱动流程。
- 未来可能增加的其他 ASR 服务实现。

统一接口的标准输出应包含：

- 输入音频路径和格式。
- ASR 服务名称、模型名称和参数摘要。
- ASR 完整识别文本。
- 字符级或 token 级时间戳列表。
- 可选置信度。
- 可追踪的 source 和 index 字段。

这种设计可以让后续替换模型、增加云端 ASR、增加本地 mock 测试时不影响对齐与合并主流程。

当前 `paraformer-zh` 服务实现策略：

- 使用 `funasr.AutoModel`。
- 默认模型目录为 `/Users/sakana/PyEnv/paraformer`。
- 默认设备为 `mps`。
- 设置 `disable_update=True`，避免每次初始化时检查 FunASR 更新。
- 将 FunASR 返回的文本和 timestamp 列表转换为项目标准 `WordTimeline`。
- 当文本包含标点但 timestamp 不包含标点时，忽略标点后再建立 token。
- 当少量连续英文或数字片段被 FunASR 合并到单个 timestamp 时，将对应片段合并为多字符 ASR token；下游对齐仍会展开 token 文本。
- 标准输出中 `provider` 和 token `source` 均使用 `paraformer-zh`。

当前已将一套真实 `paraformer-zh` 推理结果固定到 `tests/fixtures/stage1_paraformer/`。该 fixture 包含 `.mp3` 输入、匹配稿件、ASR fuzzy 输出 JSON 和 `sentence_timeline.srt`，可作为 ASR fuzzy 匹配逻辑和渲染逻辑的稳定输入。

## 音频格式策略

需求上需要支持 `.mp3`、`.wav`、`.ogg` 等常见音频格式，并统一转换为 MP3 后送入 ASR。

阶段策略：

- 当前实现：MP3 直接进入 ASR，所有非 MP3 输入由 `ffmpeg` 自动转换为 MP3；音频时长统一由 `ffprobe` 读取。
- 转换结果使用源文件指纹命名并原子落盘，有效缓存可复用；转换详情在后续处理前写入 `audio_conversion.json`。

音频转换作为独立适配层，不混入文本对齐逻辑。

## 对齐策略

最终句子时间轴应以原始稿件为准，而不是直接采用 ASR 转写文本。难点在于 ASR 输出可能出现：

- 漏字。
- 多识别出口头词或额外词。
- 替换字词。
- 标点差异。
- 英文大小写和空格差异。
- 数字读法差异，例如 `2026` 和 `二零二六`。

当前仍不做完整数字归一化、领域词和同义词处理。基础归一化后会执行顺序全局对齐；句子级 fuzzy 匹配阶段额外提供轻量数字读法兼容，例如稿件 `12元` 可与 ASR 的「十二元」对齐。该兼容只用于选择 ASR 时间窗口，不改变最终字幕文本。

当前实现基于业务前提进行收敛：音频主要由 ground truth 稿件文本生成，因此不优先处理真实对话中的大量口头插入、乱序、多人说话或长距离误配问题。ASR fuzzy 分支让每个 ground truth 分句以可解释的 fuzzy 方式匹配到 ASR token timeline，并保证最终句子时间范围不重叠；该流程已通过 `sentence_matching.py` 和 `merge.py` 实现。Qwen3 forced 分支直接对分句后的完整稿件文本和 TTS 音频做强制对齐，再按归一化 offset 回填每个分句。默认 `hybrid` 流程同时运行两条分支，最终时间以 Qwen3 forced alignment 为主，ASR fuzzy 作为 diagnostics/telemetry 保留。

推荐流程：

1. 读取 `.txt` 稿件并保留段落信息。
2. 使用可替换的句子切分接口切分稿件，或读取人工编辑后的分句文本。
3. 使用 `regex`、`hanlp`、`jieba-subtitle` 或 `llm` 作为当前分句实现。
4. 对稿件文本、ASR 文本和 forced aligner 输出进行基础归一化。
5. 在归一化后的全文级别做顺序全局对齐。
6. ASR fuzzy 分支将每个稿件句子按顺序窗口 fuzzy 匹配到 ASR token 时间范围。
7. Qwen3 forced 分支将 forced units 按归一化 offset 映射到分句范围。
8. 使用所选时间轴策略生成最终句子时间，并修正相邻句子的重叠范围。
9. 对低置信度、漏配、额外 ASR 内容和双分支时间差异生成诊断报告。

全文顺序对齐用于建立基础可检查结果。ASR fuzzy 分支已在句子层增加顺序窗口 fuzzy 匹配：每个句子只从上一句结束 token 之后向后搜索有限窗口，不做全局自由搜索，也不默认引入动态规划。ASR fuzzy 时间使用选中候选窗口的完整 token 范围，字符级完全匹配 token 只作为诊断字段保留。这样可以利用 ground truth 音频的顺序稳定性，同时容忍 ASR 局部漏字、错字、数字读法差异或额外 token。

## 句子切分策略

句子切分通过统一接口实现，当前内置三个实现：

- `regex`：按段落和强标点切分，适合保留自然句。
- `jieba-subtitle`：先按段落和强标点形成基础范围，再使用 jieba 分词拼接短句，适合短视频字幕场景。
- `llm`：通过 OpenAI-compatible Chat Completions 做短视频口播风格分句，输出纯文本换行，不使用 XML/JSON。

共同的强边界包括：

- `。`
- `！`
- `？`
- `；`
- 段落换行

`jieba-subtitle` 会把逗号、顿号、句号、问号、感叹号、分号等标点作为短语边界，但最终分句文本不保留标点。默认目标是单句归一化文本不超过 10 个字符，但不会把同一个 jieba 词切到两个分句里。如果某个词本身超过目标长度，则保留该词完整性，允许该分句超过目标长度。当前实现还内置少量短视频字幕常见软切分点，例如 `特别`、`直接`，用于得到更自然的语义短语。

当前短视频主流程可显式选择 `llm` 分句：程序先确定性隔离保护段，再按 block 并发调用 OpenAI-compatible Chat Completions 接口。每个 block 独立校验、重试和冻结成功结果；失败 block 默认交给 HanLP fallback。最终 SRT 的首尾音频对齐、短间隙填充和短字幕延长属于渲染后处理，不改变主时间轴和对齐诊断。

`llm` 分句会要求模型只在原文中插入换行，必须保留标点用于完整性校验。每个 block 内输出行直接拼接后必须与原文完全一致。校验通过后，程序再从原稿切片并去掉分句两端边界标点。当前 prompt 和本地校验要求折算长度优先为 4 到 8 个汉字，硬上限为 10；汉字计 1，英文和数字每两个计 1，标点不计，超长会触发 LLM 反馈重试。

稿件中可以使用成对保护标记跳过自动分句：

```text
[[NO_SPLIT]]这部分。不会被切开！会整体保留。[[/NO_SPLIT]]
```

标记本身会在进入归一化、匹配和字幕输出前移除，保护内容会作为一个完整分句输出，`boundary` 为 `protected`。

项目还支持分句单独运行，输出一行一句、空行分段的 `editable_segments.txt`。人工编辑后，可通过 `--segments` 读取该文件，替代自动分句结果继续执行 ASR 后的匹配、合并和渲染流程。

## 文本归一化策略

当前归一化范围：

- 去除用于对齐的标点。
- 压缩或移除空白。
- 全角和半角归一。
- 英文转小写。
- 保留原文 offset 与归一化 offset 的映射。

暂不处理：

- 阿拉伯数字与中文读法互转。
- 日期、百分比、货币和单位归一化。
- 领域词、产品名、缩写和同义词。

保留 offset 映射非常重要，因为最终输出文本、句子范围和诊断信息都应能追溯回原始稿件。

## 模糊匹配工具

成熟 Python 选项：

- `difflib.SequenceMatcher`：Python 标准库，适合建立顺序匹配和可解释诊断。
- `RapidFuzz`：高性能模糊匹配库，可用于后续补充局部评分、相似度诊断和边界优化。

当前优先使用标准库方案，降低依赖和实现复杂度。ASR fuzzy 分支使用 `difflib.SequenceMatcher` 对候选窗口评分。后续如果发现边界精度不足，再引入 RapidFuzz 或自定义动态规划对齐。

## ASR Fuzzy 匹配与时间合并策略

当前已新增独立的句子匹配模块，用于将 `SentenceSegment` 顺序匹配到 ASR token timeline。

单句匹配流程：

1. 使用句子的归一化文本作为查询。
2. 从上一句结束 token 之后开始搜索。
3. 按句子长度构造有限候选窗口。
4. 使用 `SequenceMatcher(..., autojunk=False).ratio()` 计算候选窗口与句子的相似度。
5. 选择最高分候选，并在候选内部做字符级对齐用于诊断。
6. 使用选中的 ASR 候选窗口完整 token 范围计算句子原始时间范围，避免替换字或数字读法差异导致时间轴缺口。

最终时间范围需要增加硬约束：

- `sentence_timeline.json` 中句子按 index 排列后，相邻句子时间范围不得重叠。
- 如果当前句原始开始时间早于上一句最终结束时间，则将当前句最终开始时间调整到上一句最终结束时间。
- 保留 `raw_start_ms`、`raw_end_ms` 和 `time_adjusted`，便于人工复核时间修正。
- 相邻句之间的自然停顿不需要填平。

匹配阶段会做轻量数字读法归一化，例如 `12元` 可与 ASR 的 `十二元` 对齐。该归一化只用于候选评分和时间轴选择，不改变最终稿件文本或字幕文本。

这种策略比复杂全局搜索更符合当前素材来源：它利用稿件生成音频的顺序稳定性，同时保留足够诊断字段处理少量 ASR 偏差。

## 字幕渲染策略

当前已实现统一渲染接口，首个实现为 `SrtTimelineRenderer`：

- 输入为最终 `SentenceTimelineItem` 列表。
- 输出为 `sentence_timeline.srt`。
- 字幕正文使用稿件分句文本，不使用 ASR 文本。
- 时间使用最终 `start_ms` 和 `end_ms`，因此继承无重叠修正结果。
- 缺失时间或非法时间范围的句子不会被渲染成字幕块。

后续可在同一接口下增加 VTT、CSV 或其他业务格式。

参考：

- https://docs.python.org/3/library/difflib.html
- https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html

## 其他强制对齐工具

还有一些成熟工具可以作为参考，但不适合作为当前主路径：

- Aeneas：适合文本片段与音频同步并导出字幕，但中文工作流不如 FunASR 直接。
- Montreal Forced Aligner：音素和词级强制对齐能力强，但安装和声学、发音资源要求较重。
- WhisperX：可通过 VAD 和强制对齐生成词级时间戳，但中文对齐通常需要额外模型支持，流程也更复杂。

这些工具可作为后续备选方案；当前主路径是本地 Qwen3 forced aligner 与统一 ASR 接口下本地 `paraformer-zh` 服务组成的 `hybrid` 流程。

参考：

- https://github.com/readbeyond/aeneas
- https://montreal-forced-aligner.readthedocs.io/
- https://github.com/m-bain/whisperX
