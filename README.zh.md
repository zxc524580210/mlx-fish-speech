# MLX Fish-Speech

[English](README.md) | **简体中文**

基于 [MLX](https://github.com/ml-explore/mlx)、面向 **Apple Silicon** 的 Fish-Speech 语音合成。支持 **Fish Audio S1-mini** 与 **S2 Pro** 权重（本地 `weights/`）、可选 **INT8/INT4** 量化，以及 S2 Pro 路径下的 **零样本音色克隆**。

## 运行环境

- **硬件**：Apple Silicon Mac（MLX）。
- **Python**：3.10+（见 `pyproject.toml`）。
- **权重**：按 **示例脚本** 将上游 checkpoint 放到 `weights/`（本仓库不包含权重文件）。

## 安装

```bash
pip install mlx-fish-speech
```

从源码安装：

```bash
git clone https://github.com/user/mlx-fish-speech.git
cd mlx-fish-speech
pip install -e .
```

## 快速开始

```python
from mlx_fish_speech import FishSpeech

tts = FishSpeech.from_pretrained("fishaudio/openaudio-s1-mini")
audio = tts.generate("你好，世界！")
audio.save("output.wav")
```

## 命令行

```bash
mlx-fish-speech --text "Hello, world!" --output hello.wav
mlx-fish-speech --text "今天天气真好" --output weather.wav
```

## 示例脚本

端到端脚本（从磁盘加载模型、DAC 解码、写出 WAV）：

| 脚本 | 模型 | 权重目录 |
|------|------|----------|
| `examples/s1_mini_tts.py` | S1-mini | `weights/openaudio-s1-mini/` |
| `examples/s2_pro_tts.py` | S2 Pro | `weights/s2-pro/` |

```bash
python examples/s1_mini_tts.py -t "你好世界"
python examples/s2_pro_tts.py -t "你好世界"              # 默认 INT8
python examples/s2_pro_tts.py -t "你好世界" -q int4      # 更快
python examples/s2_pro_tts.py -t "你好" --prompt your_prompt.npy   # 音色克隆
```

请从 Hugging Face 下载对应权重（[s1-mini](https://huggingface.co/fishaudio/s1-mini)、[s2-pro](https://huggingface.co/fishaudio/s2-pro)）放入上述目录。

## 上游与文档

- **代码与模型**：[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)。本仓库为 **MLX 移植**，不是上游 PyTorch / 服务端栈。
- **官方文档**：[speech.fish.audio](https://speech.fish.audio/) — [安装](https://speech.fish.audio/install/)、[命令行推理](https://speech.fish.audio/inference/#command-line-inference)、WebUI、Docker 等。

模型能力与**许可**以上游及各自模型卡为准。

## 模型说明（与官方一致）

### Fish Audio S1-mini

- **定位**：[Fish Audio S1-mini](https://huggingface.co/fishaudio/s1-mini) 为 **S1（约 4B，完整版为专有模型）的蒸馏版，约 0.5B 参数**；在 **200 万小时以上** 多语言音频上训练，并采用 **在线 RLHF** 对齐。
- **权重与许可**：**[fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini)**；模型卡注明 **CC-BY-NC-SA-4.0**（用途以许可证全文为准）。
- **能力摘要**：多语言 TTS；情感 / 语气 / 副语言标记（多为 `(emotion)`、`(tone)` 等），详见模型卡 **Emotion and Tone Support**。
- **在本仓库中**：`weights/openaudio-s1-mini` → **ContentSequence interleave**；基础合成**无需参考音频**。

### Fish Audio S2 Pro

- **定位**：[fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)，约 **4B** 慢自回归 + **Dual-AR**，RVQ **10 码本、约 21 Hz**，**Fast AR 约 4 亿参数**；官方训练规模 **1000 万小时以上**、**80+ 语言**、**GRPO** 等对齐（[S2 技术报告](https://arxiv.org/abs/2603.08823)）。
- **权重与许可**：[Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)；商业需联系 Fish Audio（见模型卡）。
- **能力摘要**：文中 **`[标签]`** 控制韵律与情感；**短时参考音频零样本克隆**（官方常见约 **10–30 秒**）；**多说话人**（`<|speaker:i|>` 等）；**多轮上下文**。上游有 **SGLang** 流式栈；本仓库为 **MLX 本地推理**，API 与性能数字与上游不同。
- **在本仓库中**：`weights/s2-pro` → **Conversation / chat template**；`--prompt` 参考码本克隆；完整标签表见 [fish-speech README](https://github.com/fishaudio/fish-speech) 与模型卡。

## 仓库结构

```
mlx-fish-speech/
├── mlx_fish_speech/
│   ├── models/           # Dual-AR（text2semantic）、DAC、RVQ 等
│   ├── utils/            # 音频 I/O、tokenizer 辅助、mlx_rvq 等
│   ├── generate.py       # MLXGenerator
│   ├── tokenizer.py
│   ├── cli.py
│   ├── conversation.py
│   ├── content_sequence.py
│   ├── voice_clone_generate.py
│   └── …
├── examples/             # s1_mini_tts.py、s2_pro_tts.py 等
├── weights/              # 本地 checkpoint（自备）
└── tests/
```

## Performance

**Apple M4 Max**、**未开启**「高性能」电源模式（默认策略）。**同一句中文**（约二十余字）、**无参考音色**，各脚本单次运行；**不含**首次加载权重。**RTF** = 纯生成耗时 ÷ 输出音频时长（小于 1 表示快于实时）；**tok/s** 为慢解码语义步吞吐。

| 模型 | 量化 | RTF | 语义 tok/s | 音频 frames/s |
|------|------|-----|------------|---------------|
| S1-mini | 无量化 | ~0.37 | ~58 | ~57 |
| S1-mini | INT8 | ~0.19 | ~117 | ~115 |
| S1-mini | INT4 | ~0.16 | ~136 | ~134 |
| S2-Pro | INT8 | ~0.76 | ~29 | ~29 |
| S2-Pro | INT4 | ~0.68 | ~32 | ~31 |

其它机型未复测；随机型、电源模式、文本与负载不同，上表仅供同条件对照。

## 法律免责声明

- 本软件按「现状」提供，作者与贡献者不对因使用或无法使用造成的损害承担责任。
- 须遵守所在地法律（著作权、肖像权、隐私、合成媒体等）。**禁止**未经授权仿声、欺诈、骚扰、诽谤等违法用途。
- 上游代码与权重另有许可（如 **FISH AUDIO RESEARCH LICENSE**）。使用或再分发权重前请阅读上游 [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) 与各模型卡。

## 语音克隆与使用责任

因**声音克隆、模仿或生成语音**产生的一切后果由**使用者本人**承担；仅在合法授权或有权使用的素材上使用相关功能。

## License

本仓库代码：**Apache 2.0**。模型权重以上游与 Hugging Face 等发布条款为准。
