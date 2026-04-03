# MLX Fish-Speech

**English** | [简体中文](README.zh.md)

Fish-Speech text-to-speech on [MLX](https://github.com/ml-explore/mlx) for **Apple Silicon**. Supports **Fish Audio S1-mini** and **S2 Pro** checkpoints (local `weights/`), optional **INT8/INT4** quantization, and **zero-shot voice cloning** on the S2 Pro path.

## Requirements

- **Hardware**: Apple Silicon Mac (MLX).
- **Python**: 3.10+ (see `pyproject.toml`).
- **Weights**: Place upstream checkpoints under `weights/` as described in **Examples** (not bundled with this repo).

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

## Quick start

```python
from mlx_fish_speech import FishSpeech

tts = FishSpeech.from_pretrained("fishaudio/openaudio-s1-mini")
audio = tts.generate("你好，世界！")
audio.save("output.wav")
```

## CLI

```bash
mlx-fish-speech --text "Hello, world!" --output hello.wav
mlx-fish-speech --text "今天天气真好" --output weather.wav
```

## Examples

End-to-end scripts (load models from disk, run DAC decode, write WAV):

| Script | Model | Weights path |
|--------|--------|----------------|
| `examples/s1_mini_tts.py` | S1-mini | `weights/openaudio-s1-mini/` |
| `examples/s2_pro_tts.py` | S2 Pro | `weights/s2-pro/` |

```bash
python examples/s1_mini_tts.py -t "你好世界"
python examples/s2_pro_tts.py -t "你好世界"              # default INT8
python examples/s2_pro_tts.py -t "你好世界" -q int4      # faster
python examples/s2_pro_tts.py -t "你好" --prompt your_prompt.npy   # voice clone
```

Download matching weights from Hugging Face ([s1-mini](https://huggingface.co/fishaudio/s1-mini), [s2-pro](https://huggingface.co/fishaudio/s2-pro)) into those directories.

## Upstream & documentation

- **Code / models**: **[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)** — this repo is a **MLX port**, not the upstream PyTorch / server stack.
- **Official guides**: **[speech.fish.audio](https://speech.fish.audio/)** — [Installation](https://speech.fish.audio/install/), [CLI inference](https://speech.fish.audio/inference/#command-line-inference), WebUI, Docker, etc.

Capabilities and **licenses** are defined upstream and on each model card.

## Models (aligned with official docs)

### Fish Audio S1-mini

- **Role**: [Fish Audio S1-mini](https://huggingface.co/fishaudio/s1-mini) — **~0.5B distilled** from **S1 (~4B; full-size proprietary)**. Trained on **2M+ hours** of multilingual audio with **online RLHF**.
- **Weights & license**: **[fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini)**; model card: **CC-BY-NC-SA-4.0** (read the full license for your use case).
- **Capabilities**: Multilingual TTS; emotion / tone / paralinguistic markers (often `(emotion)`, `(tone)` — see **Emotion and Tone Support** on the card).
- **In this repo**: `weights/openaudio-s1-mini` → **ContentSequence interleave**; **no reference audio** for basic TTS.

### Fish Audio S2 Pro

- **Role**: [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) — **~4B** slow AR + **Dual-AR**, RVQ **10 codebooks ~21 Hz**, **Fast AR ~400M**. Training scale: **10M+ hours**, **80+ languages**, **GRPO**-style alignment — [S2 technical report](https://arxiv.org/abs/2603.08823).
- **Weights & license**: **[Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)**; commercial use needs Fish Audio (see card).
- **Capabilities**: **`[tags]`** for prosody/emotion; **zero-shot cloning** (~**10–30 s** reference audio per upstream); multi-speaker (`<|speaker:i|>`, etc.); multi-turn. Upstream offers **SGLang** streaming; **this repo = MLX local inference** (APIs and benchmarks differ).
- **In this repo**: `weights/s2-pro` → **Conversation / chat template**; `--prompt` codebooks for cloning; tag lists — [fish-speech README](https://github.com/fishaudio/fish-speech) + model cards.

## Repository layout

```
mlx-fish-speech/
├── mlx_fish_speech/
│   ├── models/           # Dual-AR (text2semantic), DAC, RVQ helpers
│   ├── utils/            # audio I/O, tokenizer helpers, mlx_rvq, …
│   ├── generate.py       # MLXGenerator
│   ├── tokenizer.py
│   ├── cli.py
│   ├── conversation.py
│   ├── content_sequence.py
│   ├── voice_clone_generate.py
│   └── …
├── examples/             # s1_mini_tts.py, s2_pro_tts.py, …
├── weights/              # local checkpoints (you provide)
└── tests/
```

## Performance

**Apple M4 Max**, **High Power Mode off** (default macOS power). Same **Chinese sentence** (~20+ chars), **no reference voice**, one run per script; **excludes** cold load. **RTF** = generation time ÷ audio length (below 1 = faster than real time); **tok/s** = slow-decoder semantic throughput.

| Model | Quantization | RTF | Semantic tok/s | Audio frames/s |
|-------|----------------|-----|----------------|----------------|
| S1-mini | none | ~0.37 | ~58 | ~57 |
| S1-mini | INT8 | ~0.19 | ~117 | ~115 |
| S1-mini | INT4 | ~0.16 | ~136 | ~134 |
| S2-Pro | INT8 | ~0.76 | ~29 | ~29 |
| S2-Pro | INT4 | ~0.68 | ~32 | ~31 |

Other hardware was not re-tested; numbers vary with chip, power mode, text, reference audio, and load.

## Legal disclaimer

- Software is **as-is**; authors and contributors **disclaim** liability for damages from use or inability to use it.
- Obey local laws (copyright, personality rights, privacy, synthetic media, etc.). **No** unauthorized voice imitation, fraud, harassment, or defamation.
- Upstream code and weights have separate licenses (e.g. **FISH AUDIO RESEARCH LICENSE**). Read the upstream [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) and each model card before redistributing weights.

## Voice cloning responsibility

**You** bear all legal and other consequences of **cloning, imitation, or generated speech**. Use only with proper rights or authorization to the source audio.

## License

**Apache 2.0** for **this repo’s code**. Model weights follow upstream / Hugging Face terms.
