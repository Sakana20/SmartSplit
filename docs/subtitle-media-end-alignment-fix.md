# 字幕末尾精确对齐原始媒体方案

## 1. 最终目标

SmartSplit 的末条字幕用于覆盖最终视频，因此默认目标是原始视频结束位置，而不是音频轨结束位置。

规则如下：

1. 原始媒体包含实际视频流时，选择第一个视频流 `v:0` 的呈现结束时间；忽略 MP3/M4A 封面图等 `attached_pic` 流。
2. 原始媒体不含视频流时，选择第一个音频流 `a:0` 的呈现结束时间。
3. 目标四舍五入到最近毫秒后写入 SRT。
4. 不再固定按 30fps 向上取整。
5. 为 ASR 或 forced aligner 转码生成的 MP3 不参与字幕末尾计算。

这里的原始媒体为 `--subtitle-alignment-audio` 指定的文件；未指定时使用 `--audio`。参数名因兼容历史调用仍保留 `audio`，但它可以接收 MP4 等复合媒体。

## 2. 原问题

旧实现执行：

```text
ffprobe format.duration → ceil(duration × 30) / 30 → SRT 末尾
```

这会把任何媒体结束时间向上吸附到固定 30fps，产生 0–33ms 的额外延后。输入视频为 24fps、25fps、29.97fps 或其他帧率时，固定 30fps 也没有正确的编辑语义。

本次 TBFC 样本均为 25fps。例如 TBFC-005：

```text
视频流结束：18.720s
音频流结束：18.645s
旧 SRT 结束：18.733s
正确 SRT 结束：18.720s
```

正确目标是视频流的 `18.720s`。音频较短不应导致末条字幕提前消失；旧结果 `18.733s` 则是固定 30fps 取整额外增加了 13ms。

## 3. 为什么显式选择视频流

TBFC 样本的 `format.duration` 恰好等于视频流长度，但实现仍应显式选择视频流，因为容器总时长可能受其他音轨、数据流、时间戳或封装方式影响。需求是“对齐视频”，因此目标对象必须是视频流，而不是碰巧相等的容器字段。

纯音频文件没有视频流，才合理地使用音频流结束时间。

## 4. 呈现结束时间

目标流结束位置按媒体呈现时间轴计算：

```text
normalized_start = stream_start - media_start
stream_end = normalized_start + stream_duration
```

实现优先使用：

```text
duration_ts × time_base
```

这样可以用整数时间戳和有理数 time base 计算，避免二进制浮点误差。流级整数时间不可用时，依次回退到：

1. 流级 `duration`；
2. 目标流最后一个 packet 的 `pts_time + duration_time`；
3. 仅单流媒体允许使用 `format.duration`。

多流媒体无法确定目标流结束时间时必须明确失败，不能静默改用另一条流或容器总时长。

## 5. 与视频帧率的关系

方案不需要额外识别 24fps、25fps 或 30fps 再做一次取整。视频流的 `duration_ts × time_base` 已经给出该视频在媒体时间轴上的真实结束位置。

SRT 支持毫秒时间戳，所以只需量化到最近毫秒：

```text
18.720000s → 00:00:18,720
```

这避免了旧实现将 25fps 视频的 `18.720s` 再向上推到 30fps 边界 `18.733s`。

## 6. Pipeline 语义

Python API 使用显式开关区分“未指定路径”和“关闭对齐”：

```python
run_pipeline(
    ...,
    subtitle_alignment_audio: Path | None = None,
    align_last_subtitle_to_audio_end: bool = True,
)
```

解析规则：

```text
align_last=false → 不探测、不修正
align_last=true 且显式给路径 → 使用 subtitle_alignment_audio
align_last=true 且未给路径 → 使用 audio_path
```

CLI、SmartSplit skill 入口和 Python API 使用相同默认行为。

## 7. 渲染诊断

`subtitle_render_report.json.end_alignment` 记录：

```json
{
  "enabled": true,
  "applied": true,
  "media_path": "/path/to/input.mp4",
  "media_format_duration_ms": 18720,
  "target_stream_type": "video",
  "target_stream_index": 0,
  "target_stream_start_ms": 0,
  "target_stream_end_ms": 18720,
  "target_stream_duration_ms": 18720,
  "timing_source": "stream_duration_ts",
  "original_last_cue_end_ms": 18420,
  "rendered_last_cue_end_ms": 18720,
  "quantization": "nearest_millisecond"
}
```

纯音频输入的 `target_stream_type` 为 `audio`。

## 8. 测试与验收

自动化测试至少覆盖：

- 视频长于音频的 MP4：选择视频流。
- 音频长于视频的 MP4：仍选择视频流。
- 纯 WAV、MP3、OGG：没有视频时选择音频流。
- 非零 stream/media 起点的时间轴归一化。
- 流 duration 缺失时的 packet 回退。
- 多流媒体无法取得目标流结束时明确失败。
- CLI、Python API 默认和关闭行为一致。
- 报告记录目标流类型、index、来源和最终毫秒值。

真实 TBFC 回归的验收值为：SRT 最后一条结束时间等于第一个视频流呈现结束时间的最近毫秒。以 TBFC-005 为例，必须为 `18.720s`，不能是音频流的 `18.645s`，也不能是旧 30fps 取整结果 `18.733s`。
