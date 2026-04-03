#!/usr/bin/env python
"""
S1-mini TTS — 基本用法示例

对应上游 **Fish Audio S1-mini**（Hugging Face: https://huggingface.co/fishaudio/s1-mini ）：
约 0.5B、S1 全尺寸版的蒸馏模型；多语言、官方提供大量 (情感)/(语气) 类文本标记说明（见模型卡）。
许可以模型卡为准（CC-BY-NC-SA-4.0）。

本仓库内为该权重的 MLX 实现；本示例从 `weights/openaudio-s1-mini` 加载，使用 **ContentSequence interleave**
格式，无需参考音频即可合成。架构规模随权重而定（例如约 28 层 / 1024 dim 量级，以 checkpoint 为准）。

用法:
    python examples/s1_mini_tts.py
    python examples/s1_mini_tts.py -t "你好世界" -q int8
    python examples/s1_mini_tts.py --text "Hello world" --quantize int4
"""
import sys, os, time, argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import mlx.core as mx, mlx.nn as nn
import soundfile as sf
from mlx_fish_speech.generate import MLXGenerator
from mlx_fish_speech.models.dac import DACCodec

MODEL = str(_REPO / "weights" / "openaudio-s1-mini")
OUTPUT_DIR = str(_REPO / "outputs")


def main():
    parser = argparse.ArgumentParser(description="S1-mini TTS 语音合成")
    parser.add_argument("-t", "--text", default="你好，欢迎使用语音合成系统。今天天气真不错。")
    parser.add_argument("-q", "--quantize", choices=["none", "int8", "int4"], default="none",
                        help="量化模式 (默认: none)")
    parser.add_argument("-o", "--output", default=None, help="输出 WAV 路径")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 50)
    print("🐟 S1-mini TTS")
    print("=" * 50)

    # 1. 加载模型
    gen = MLXGenerator(MODEL)

    # 2. 量化（可选）
    if args.quantize != "none":
        bits = 8 if args.quantize == "int8" else 4
        nn.quantize(gen.model, bits=bits, group_size=64,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and
                    m.weight.shape[0] >= 256 and m.weight.shape[1] >= 256)
        mx.eval(gen.model.parameters())
        print(f"[{args.quantize.upper()} quantized]")

    dac = DACCodec.from_pretrained(MODEL)

    # 3. 生成
    print(f"\nText: {args.text}")
    print("Generating...\n")

    t0 = time.time()
    codes = gen.generate(
        args.text,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_time = time.time() - t0

    # 4. 解码为音频
    codes_mx = mx.array(codes.astype(np.int32)) if isinstance(codes, np.ndarray) else codes
    audio, _ = dac.decode(codes_mx)
    mx.eval(audio)
    a = np.array(audio).squeeze()

    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_DIR, "s1mini_output.wav")
    sf.write(output_path, a, 44100)

    print(f"Audio: {len(a)/44100:.2f}s")
    print(f"Speed: {codes.shape[1]/gen_time:.1f} frames/s")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
