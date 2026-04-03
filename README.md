# MLX Fish-Speech

Fish-Speech 文本转语音的 MLX 实现，面向 Apple Silicon。

## 上游与参考

算法与模型设计请参考官方仓库：

**[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)**

本仓库为在 Apple 芯片上使用 [MLX](https://github.com/ml-explore/mlx) 的移植与工具代码，与上游的 PyTorch / 服务端推理栈不同；使用与排障请以本仓库说明为准，模型能力与许可以上游与权重发布页为准。

## Features

- Native MLX implementation for Apple Silicon
- Strong Chinese TTS quality (CER ~1.3% in upstream benchmarks)
- Zero-shot voice cloning support
- Real-time synthesis on M1/M2/M3/M4 chips

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

在 **macOS 已开启高性能（或接电满血）电源策略** 的前提下，用本仓库 `examples/s2_pro_tts.py`、**同一句中文**（约二十余字、**无参考音色**）对 **S2-Pro** 做的单次实测（不含首次加载权重时间）。**RTF** = 纯生成耗时 ÷ 输出音频时长（小于 1 表示快于实时）；**tok/s** 为慢解码语义步吞吐。

| 芯片 / 配置 | 量化 | RTF | 语义 tok/s | 音频 frames/s |
|-------------|------|-----|------------|---------------|
| Apple M4 Max | INT8 | ~0.74 | ~29 | ~29 |
| Apple M4 Max | INT4 | ~0.67 | ~32 | ~32 |

其它机型（M3 Pro、M1 等）未在同一环境复测；随机型、量化、文本长度、是否带音色参考及系统负载变化较大，上表仅作同条件对照。

## 法律免责声明

- 本软件按「现状」提供，作者与贡献者**不对**因使用或无法使用本软件造成的任何直接、间接、偶然或后果性损害承担责任。
- 使用者须自行遵守所在地关于著作权、肖像权、个人信息保护、深度伪造（deepfake）、电信与内容监管等法律法规。**禁止**将本工具用于未经授权模仿他人声音、欺诈、骚扰、诽谤或其他违法用途。
- 上游 Fish Speech 项目对其代码与关联权重另有许可（例如 **FISH AUDIO RESEARCH LICENSE**），使用权重与再分发前请阅读上游 [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) 与模型卡说明。

## 语音克隆与使用责任

使用本仓库进行**声音克隆、模仿特定人声或生成语音内容**时，所产生的一切法律后果、民事或行政责任以及对第三方造成的损害，**均由使用者本人承担**；使用者应仅在取得合法授权或仅针对本人/有权使用的素材的前提下使用相关功能。

## License

Apache 2.0（本仓库代码）。模型权重许可以上游与 Hugging Face 等平台发布为准。
