#!/usr/bin/env python
"""
S2-Pro TTS — 基本用法示例（含音色克隆）

Fish-Speech S2-Pro: 36 层, 2560 dim, ~9.12GB
使用 Conversation chat template 格式，支持音频参考（音色克隆）和情感标签。

用法:
    # 基本生成（无音色参考）
    python examples/s2_pro_tts.py -t "你好世界"

    # 带音色克隆
    python examples/s2_pro_tts.py -t "你好世界" --prompt haiyan_prompt.npy

    # INT8 量化 + 情感标签
    python examples/s2_pro_tts.py -t "[开心]今天天气真好！" --prompt haiyan_prompt.npy -q int8

    # INT4 量化（最快）
    python examples/s2_pro_tts.py -t "这是一段测试" -q int4

情感标签示例: [开心] [悲伤] [愤怒] [温柔] [焦急] [兴奋] [大笑] [耳语] [大喊]
"""
import sys, os, time, argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import mlx.core as mx, mlx.nn as nn
import soundfile as sf

from mlx_fish_speech.tokenizer import FishTokenizer
from mlx_fish_speech.generate import MLXGenerator
from mlx_fish_speech.models.dac import DACCodec
from mlx_fish_speech.voice_clone_generate import (
    DEFAULT_REFERENCE_TRANSCRIPT,
    generate_with_reference,
)

MODEL = str(_REPO / "weights" / "s2-pro")
WEIGHTS_DIR = str(_REPO / "weights")
OUTPUT_DIR = str(_REPO / "outputs")


def main():
    if not Path(MODEL).is_dir():
        raise SystemExit(
            f"Missing S2-Pro weights at {MODEL}. Place the checkpoint under weights/s2-pro/."
        )
    parser = argparse.ArgumentParser(description="S2-Pro TTS 语音合成")
    parser.add_argument("-t", "--text",
                        default="你好，这是一段由Fish Speech S2 Pro模型生成的语音。")
    parser.add_argument("--prompt", default=None,
                        help="音色参考文件 (在 weights/ 目录下，如 haiyan_prompt.npy)")
    parser.add_argument(
        "--ref-text",
        default=DEFAULT_REFERENCE_TRANSCRIPT,
        metavar="TEXT",
        help="与参考音频一致的口播转写（带 --prompt 时必填准，否则听感异常）",
    )
    parser.add_argument("-q", "--quantize", choices=["none", "int8", "int4"], default="int8",
                        help="量化模式 (默认: int8)")
    parser.add_argument("-o", "--output", default=None, help="输出 WAV 路径")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    print("=" * 50)
    print("🐟 S2-Pro TTS")
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

    print(f"\nText: {args.text}")
    print(f"Prompt: {args.prompt or '(无 — 使用默认音色)'}")
    print("Generating...\n")

    mx.set_cache_limit(4 * 1024 * 1024 * 1024)
    t0 = time.time()

    # 3. 生成
    if args.prompt:
        # 带音色克隆
        tokenizer = FishTokenizer.from_pretrained(MODEL)
        prompt_codes = np.load(os.path.join(WEIGHTS_DIR, args.prompt))
        codes = generate_with_reference(
            gen,
            tokenizer,
            args.text,
            prompt_codes,
            args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        # 无参考
        codes = gen.generate(
            args.text,
            temperature=args.temperature,
            top_p=args.top_p)

    gen_time = time.time() - t0

    # 4. 解码为音频
    audio, _ = dac.decode(mx.array(codes))
    mx.eval(audio)
    a = np.array(audio).squeeze()

    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_DIR, "s2pro_output.wav")
    sf.write(output_path, a, 44100)

    print(f"Audio: {len(a)/44100:.2f}s")
    print(f"Frames: {codes.shape[1]} | Speed: {codes.shape[1]/gen_time:.1f} frames/s")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
