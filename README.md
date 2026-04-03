# MLX Fish-Speech

**English** | [简体中文](README.zh.md)

Fish-Speech text-to-speech running on [MLX](https://github.com/ml-explore/mlx) for Apple Silicon.

## Upstream

Algorithms and model design follow the official project:

**[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)**

This repository is a MLX port and toolkit for Apple chips. It is **not** the upstream PyTorch / server stack; refer to this repo for MLX usage and troubleshooting. Model capabilities and licensing are defined by upstream and each model’s release page.

Full upstream install, CLI / WebUI / Docker, and server docs: **[speech.fish.audio](https://speech.fish.audio/)** (e.g. [Installation](https://speech.fish.audio/install/), [CLI inference](https://speech.fish.audio/inference/#command-line-inference)).

## Models (aligned with official docs)

### Fish Audio S1-mini

- **Role**: [Fish Audio S1-mini](https://huggingface.co/fishaudio/s1-mini) is a **~0.5B distilled** variant of **S1 (~4B; full-size model is proprietary)**. Trained on **2M+ hours** of multilingual audio with **online RLHF**.
- **Weights & license**: **[fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini)** on Hugging Face; model card states **CC-BY-NC-SA-4.0** (see the full license for commercial and other uses).
- **Capabilities**: Multilingual TTS; the card documents many **emotion / tone / paralinguistic** markers (often `(emotion)`, `(tone)`, etc.). See **Emotion and Tone Support** on the model card.
- **In this repo**: Examples load from `weights/openaudio-s1-mini` (maps to upstream `fishaudio/s1-mini`); **ContentSequence interleave** format; **no reference audio** required for basic synthesis.

### Fish Audio S2 Pro

- **Role**: Flagship multilingual TTS ([fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)): **~4B** slow autoregressive backbone + **Dual-AR** with RVQ audio codec (**10 codebooks, ~21 Hz**); **Fast AR ~400M** parameters for the remaining codebooks per step. Official training scale: **10M+ hours**, **80+ languages**, **GRPO**-style RL alignment ([S2 technical report](https://arxiv.org/abs/2603.08823)).
- **Weights & license**: **[Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)**; research and non-commercial use free; commercial use requires Fish Audio (see model card).
- **Capabilities**: Fine-grained prosody/emotion via **`[tags]`** in text (open-ended descriptions, not only a fixed list); **zero-shot voice cloning** from short reference audio (official guidance often **~10–30 s**); **multi-speaker** (`<|speaker:i|>`, etc., in upstream docs); **multi-turn** context. Upstream also ships **SGLang** streaming inference; **this repo is MLX local inference**—APIs and benchmark numbers differ from upstream.
- **In this repo**: Examples load `weights/s2-pro`; **Conversation / chat template**; optional `--prompt` codebooks for cloning; tag usage follows upstream `[tag]` style—see [fish-speech README](https://github.com/fishaudio/fish-speech) and model cards for full tag lists.

## Features

- Native MLX on Apple Silicon
- Inference and optional quantization for **S1-mini** and **S2 Pro** weight layouts (see `examples/`)
- Zero-shot voice cloning on the S2 Pro path (reference audio); S1-mini follows upstream multilingual + marker behavior (see model card)
- Low-latency synthesis on Apple Silicon (see **Performance** below)

## Installation

```bash
pip install mlx-fish-speech
```

From source:

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

# Chinese text
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

Measured on **Apple M4 Max** with **High Power Mode off** (default macOS power settings). Same **Chinese sentence** (~20+ characters), **no reference voice**, one run each for `examples/s1_mini_tts.py` and `examples/s2_pro_tts.py` (S2 weights under `weights/s2-pro/`). **Excludes** first-time weight load. **RTF** = generation wall time ÷ output audio duration (below 1 means faster than real time); **tok/s** is slow-decoder semantic throughput.

| Model | Quantization | RTF | Semantic tok/s | Audio frames/s |
|-------|----------------|-----|----------------|----------------|
| S1-mini | none | ~0.37 | ~58 | ~57 |
| S1-mini | INT8 | ~0.19 | ~117 | ~115 |
| S1-mini | INT4 | ~0.16 | ~136 | ~134 |
| S2-Pro | INT8 | ~0.76 | ~29 | ~29 |
| S2-Pro | INT4 | ~0.68 | ~32 | ~31 |

Other Macs were not re-tested under the same setup; chip, power mode, quantization, text length, reference audio, and load strongly affect numbers—use the table for like-for-like comparison only.

## Legal disclaimer

- This software is provided **as-is**; authors and contributors **disclaim** liability for any direct, indirect, incidental, or consequential damages from use or inability to use it.
- You must comply with local laws on copyright, personality rights, privacy, synthetic media, telecom, and content regulation. **Do not** use this tool for unauthorized voice imitation, fraud, harassment, defamation, or other unlawful purposes.
- Upstream Fish Speech code and weights have their own licenses (e.g. **FISH AUDIO RESEARCH LICENSE**). Read the upstream [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) and each model card before using or redistributing weights.

## Voice cloning responsibility

If you use this repo for **voice cloning, imitation, or speech generation**, **you alone** bear all legal, civil, and regulatory consequences and any harm to third parties. Only use these features with proper authorization or on audio you have the right to use.

## License

Apache 2.0 for **this repository’s code**. Model weights are governed by upstream and Hugging Face (or other) release terms.
