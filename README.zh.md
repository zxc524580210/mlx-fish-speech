# MLX Fish-Speech

[English](README.md) | **简体中文**

Fish-Speech 文本转语音的 MLX 实现，面向 Apple Silicon。

## 上游与参考

算法与模型设计请参考官方仓库：

**[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)**

本仓库为在 Apple 芯片上使用 [MLX](https://github.com/ml-explore/mlx) 的移植与工具代码，与上游的 PyTorch / 服务端推理栈不同；使用与排障请以本仓库说明为准，模型能力与许可以上游与权重发布页为准。

更完整的安装、命令行 / WebUI / Docker 与服务器推理说明见官方文档：[speech.fish.audio](https://speech.fish.audio/)（例如 [安装](https://speech.fish.audio/install/)、[命令行推理](https://speech.fish.audio/inference/#command-line-inference)）。

## 模型说明（与官方一致）

### Fish Audio S1-mini

- **定位**：[Fish Audio S1-mini](https://huggingface.co/fishaudio/s1-mini) 为 **S1（约 4B，完整版为专有模型）的蒸馏版，约 0.5B 参数**；在 **200 万小时以上** 多语言音频上训练，并采用 **在线 RLHF** 对齐。
- **权重与许可**：公开权重见 Hugging Face **[fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini)**；模型卡注明许可为 **CC-BY-NC-SA-4.0**（商业等用途以模型卡与许可证全文为准）。
- **能力摘要**：多语言 TTS；官方文档列出大量 **情感 / 语气 / 副语言** 标记（模型卡中为英文说明，标记多为 `(emotion)`、`(tone)` 等形式），详见模型卡 **Emotion and Tone Support**。
- **在本仓库中**：示例默认从本地目录 `weights/openaudio-s1-mini` 加载（与上游 `fishaudio/s1-mini` 对应）；使用 **ContentSequence interleave** 对话格式，**无需参考音频**即可合成。

### Fish Audio S2 Pro

- **定位**：官方旗舰多语言 TTS（[fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)），约 **4B** 慢自回归主干 + **Dual-AR**：配合 RVQ 音频编解码（**10 个码本、约 21 Hz**），其中 **Fast AR 约 4 亿参数** 逐步生成其余码本；训练规模官方表述为 **1000 万小时以上**、**80+ 语言**，并用 **GRPO** 等做强化学习对齐（详见 [S2 技术报告](https://arxiv.org/abs/2603.08823)）。
- **权重与许可**：**[Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)**；研究与非商业免费，商业需联系 Fish Audio（见模型卡）。
- **能力摘要**：文中 **`[标签]`** 做韵律与情感的细粒度控制（官方强调支持大量自由描述，不限固定词表）；支持 **短时参考音频的零样本音色克隆**（官方常见建议约 **10–30 秒**）、**多说话人**（`<|speaker:i|>` 等，见上游文档）、**多轮上下文**。上游还提供基于 **SGLang** 的流式推理栈；本仓库为 **MLX 本地推理**，API 与性能数字与上游不完全相同。
- **在本仓库中**：示例从 `weights/s2-pro` 加载；使用 **Conversation / chat template** 格式，可选 `--prompt` 参考码本做克隆；情感示例与官方 `[tag]` 用法一致，具体标签表以 [fish-speech README](https://github.com/fishaudio/fish-speech) 与模型卡为准。

## Features

- Native MLX implementation for Apple Silicon
- 支持 **S1-mini**（轻量）与 **S2 Pro**（旗舰）权重路径下的推理与量化（见 `examples/`）
- S2 Pro 路径下支持零样本音色克隆（参考音频）；S1-mini 以官方多语言与情感标记能力为基础（见模型卡）
- 在 Apple Silicon 上可做到较低延迟合成（具体见下文 Performance）

## Installation

```bash
pip install mlx-fish-speech
```

Or from source:

```bash
git clone https://github.com/user/mlx-fish-speech.git
cd mlx-fish-speech
pip install -e .
```

## Quick Start

```python
from mlx_fish_speech import FishSpeech

# Initialize model
tts = FishSpeech.from_pretrained("fishaudio/openaudio-s1-mini")

# Generate speech
audio = tts.generate("你好，世界！")
audio.save("output.wav")
```

## CLI Usage

```bash
# Generate speech from text
mlx-fish-speech --text "Hello, world!" --output hello.wav

# With Chinese text
mlx-fish-speech --text "今天天气真好" --output weather.wav
```

## Architecture

```
mlx-fish-speech/
├── mlx_fish_speech/
│   ├── models/
│   │   ├── text2semantic.py   # DualAR Transformer
│   │   ├── dac.py             # DAC Codec (Encoder/Decoder)
│   │   └── rvq.py             # Residual Vector Quantization
│   ├── utils/
│   │   ├── tokenizer.py       # Tiktoken wrapper
│   │   └── audio.py           # Audio I/O utilities
│   └── generate.py            # High-level generation API
├── examples/
└── tests/
```

## Performance

在 **Apple M4 Max**、**未开启** macOS「高性能」电源模式（系统默认电源策略）下，用 **同一句中文**（约二十余字、**无参考音色**）分别运行 `examples/s1_mini_tts.py`（S1-mini）与 `examples/s2_pro_tts.py`（S2-Pro，权重位于 `weights/s2-pro/`），单次实测（**不含**首次加载权重时间）。**RTF** = 纯生成耗时 ÷ 输出音频时长（小于 1 表示快于实时）；**tok/s** 为慢解码语义步吞吐。

| 模型 | 量化 | RTF | 语义 tok/s | 音频 frames/s |
|------|------|-----|------------|---------------|
| S1-mini | 无量化 | ~0.37 | ~58 | ~57 |
| S1-mini | INT8 | ~0.19 | ~117 | ~115 |
| S1-mini | INT4 | ~0.16 | ~136 | ~134 |
| S2-Pro | INT8 | ~0.76 | ~29 | ~29 |
| S2-Pro | INT4 | ~0.68 | ~32 | ~31 |

其它机型未在同一环境复测；随机型、是否开启「高性能」、量化、文本长度、是否带音色参考及系统负载变化较大，上表仅作同条件对照。

## 法律免责声明

- 本软件按「现状」提供，作者与贡献者**不对**因使用或无法使用本软件造成的任何直接、间接、偶然或后果性损害承担责任。
- 使用者须自行遵守所在地关于著作权、肖像权、个人信息保护、深度伪造（deepfake）、电信与内容监管等法律法规。**禁止**将本工具用于未经授权模仿他人声音、欺诈、骚扰、诽谤或其他违法用途。
- 上游 Fish Speech 项目对其代码与关联权重另有许可（例如 **FISH AUDIO RESEARCH LICENSE**），使用权重与再分发前请阅读上游 [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) 与模型卡说明。

## 语音克隆与使用责任

使用本仓库进行**声音克隆、模仿特定人声或生成语音内容**时，所产生的一切法律后果、民事或行政责任以及对第三方造成的损害，**均由使用者本人承担**；使用者应仅在取得合法授权或仅针对本人/有权使用的素材的前提下使用相关功能。

## License

Apache 2.0（本仓库代码）。模型权重许可以上游与 Hugging Face 等平台发布为准。
