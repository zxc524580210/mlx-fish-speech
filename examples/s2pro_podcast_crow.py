#!/usr/bin/env python
"""
S2-Pro 播客生成：乌鸦喝水故事
Speaker 0: 旁白/讲述者
Speaker 1: 乌鸦/对话者

默认与 examples/s2_pro_tts.py **不传 --prompt** 相同：走 MLXGenerator.generate()，
每句前加 <|speaker:0|> / <|speaker:1|> 区分角色，中文稳定。

若需要参考音色克隆，请加 --voice-clone，并提供 speaker0/1 的 .npy 与 --ref-text0/1
（转写须与参考音频一致，否则易乱码）。

用法: python examples/s2pro_podcast_crow.py
      python examples/s2pro_podcast_crow.py --max-segments 1
      python examples/s2pro_podcast_crow.py --voice-clone --speaker0 haiyan_prompt.npy
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

# 播客脚本：乌鸦喝水（带情感标签）
SCRIPT = [
    (0, "[温柔]大家好，欢迎收听今天的故事时间。[开心]今天我们要讲一个经典的寓言故事——乌鸦喝水。"),
    (1, "[兴奋]嗯！这个故事我从小就听过，特别有意思！快讲快讲！"),
    (0, "[娓娓道来]话说在很久很久以前，有一只乌鸦在烈日下飞了很长时间，[同情]它又渴又累，急切地想找到水喝。"),
    (1, "[焦急]天啊，好渴啊！我已经飞了好久了，嗓子都快冒烟了！[着急]快帮我找找，哪里有水呀？"),
    (0, "[悬念]就在这时，乌鸦发现了一个瓶子。[惋惜]瓶子里有一些水，但是水位很低，它的嘴巴根本够不到。"),
    (1, "[惊喜]哎呀，有水！[失望]可是……这瓶口太小了，我的嘴伸不进去。[沮丧]水在下面，我怎么才能喝到呢？"),
    (0, "[紧张]乌鸦试了好几次，把嘴伸进瓶口，但始终够不到水面。[同情]它非常着急，围着瓶子转了好几圈。"),
    (1, "[焦躁]不行不行，这样不行！[冷静]我得想个办法才行……让我想想……"),
    (0, "[惊喜]突然！乌鸦灵光一闪！[兴奋]它看到地上有很多小石子，一个绝妙的主意浮现在脑海中！"),
    (1, "[得意]对了！我可以把石子一颗一颗地放到瓶子里，这样水面不就升高了吗？[骄傲]我真是太聪明了！哈哈哈！"),
    (0, "[平稳]于是，乌鸦开始叼起一颗又一颗的石子，[赞叹]耐心地把它们放进瓶子里。随着石子越来越多，水面也慢慢地升高了。"),
    (1, "[期待]一颗、两颗、三颗……[激动]水面越来越高了！马上就能喝到了！[加油]加油加油！"),
    (0, "[欣慰]终于，水面升到了瓶口附近。[开心]乌鸦开心地喝到了水，解了渴。"),
    (1, "[大笑]哈哈哈，终于喝到水了！太开心了！[自豪]只要动脑筋，就没有解决不了的问题！"),
    (0, "[认真]这个故事告诉我们，遇到困难的时候不要轻易放弃，[语重心长]要善于观察和思考。只要开动脑筋，总能找到解决问题的方法。"),
    (1, "[认同]说得对！[感慨]我觉得这个故事最棒的地方就是告诉大家，智慧比蛮力更重要。"),
    (0, "[温暖]好了，今天的故事就讲到这里。[感谢]感谢大家的收听，我们下期再见！"),
    (1, "[开心]拜拜！下次见！"),
]

def main():
    if not Path(MODEL).is_dir():
        raise SystemExit(
            f"Missing S2-Pro weights at {MODEL}. Place the checkpoint under weights/s2-pro/."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quantize", choices=["none", "int8", "int4"], default="int8")
    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="使用参考 .npy 克隆音色（默认关闭，与 s2_pro_tts 无 --prompt 一致）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度（与 MLXGenerator.generate 默认一致）",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="nucleus 采样（与 MLXGenerator.generate 默认一致）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="每段最大生成 token 数（非 clone 模式）",
    )
    parser.add_argument(
        "--speaker0",
        default="fake.npy",
        help="--voice-clone 时：讲述者参考 tokens",
    )
    parser.add_argument(
        "--speaker1",
        default="fake.npy",
        help="--voice-clone 时：乌鸦参考 tokens",
    )
    parser.add_argument(
        "--ref-text0",
        default=DEFAULT_REFERENCE_TRANSCRIPT,
        metavar="TEXT",
        help="--voice-clone：与 speaker0 .npy 一致的口播转写",
    )
    parser.add_argument(
        "--ref-text1",
        default=DEFAULT_REFERENCE_TRANSCRIPT,
        metavar="TEXT",
        help="--voice-clone：与 speaker1 .npy 一致的口播转写",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N script lines (smoke test)",
    )
    args = parser.parse_args()

    script = SCRIPT if args.max_segments is None else SCRIPT[: args.max_segments]

    print("=" * 60)
    print("🐟 S2-Pro 播客：乌鸦喝水（情感版）")
    print("=" * 60)

    # Load model
    gen = MLXGenerator(MODEL)
    if args.quantize != "none":
        bits = 8 if args.quantize == "int8" else 4
        nn.quantize(gen.model, bits=bits, group_size=64,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and
                    m.weight.shape[0] >= 256 and m.weight.shape[1] >= 256)
        mx.eval(gen.model.parameters())
        print(f"[{args.quantize.upper()} quantized]")

    dac = DACCodec.from_pretrained(MODEL)
    tokenizer = None
    speaker_prompts = {}
    ref_texts = {0: args.ref_text0, 1: args.ref_text1}
    if args.voice_clone:
        tokenizer = FishTokenizer.from_pretrained(MODEL)
        speaker_prompts = {
            0: np.load(os.path.join(WEIGHTS_DIR, args.speaker0)),
            1: np.load(os.path.join(WEIGHTS_DIR, args.speaker1)),
        }
    speaker_names = {0: "讲述者", 1: "乌鸦"}

    all_audio = []
    total_time = 0
    mx.set_cache_limit(4 * 1024 * 1024 * 1024)

    for idx, (speaker, text) in enumerate(script):
        name = speaker_names[speaker]
        print(f"\n[{idx+1}/{len(script)}] {name}: {text[:30]}...", flush=True)

        t0 = time.time()
        if args.voice_clone:
            codes_np = generate_with_reference(
                gen,
                tokenizer,
                text,
                speaker_prompts[speaker],
                ref_texts[speaker],
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            # 与 s2_pro_tts 无 --prompt 相同：内置 chat 模板 + 多说话人标签
            line = text if "<|speaker:" in text else f"<|speaker:{speaker}|>{text}"
            codes_np = gen.generate(
                line,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=10,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        seg_time = time.time() - t0
        total_time += seg_time

        n_frames = codes_np.shape[1]
        audio, _ = dac.decode(mx.array(codes_np))
        mx.eval(audio)
        a = np.array(audio).squeeze()
        all_audio.append(a)
        print(
            f"  {len(a)/44100:.2f}s | {n_frames} frames | {n_frames/seg_time:.1f} f/s",
            flush=True,
        )

        all_audio.append(np.zeros(int(44100 * 0.3), dtype=np.float32))

    # Concatenate all audio
    final_audio = np.concatenate(all_audio)
    output_path = str(_REPO / "outputs" / "podcast_crow.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, final_audio, 44100)

    print(f"\n{'='*60}")
    print(f"✅ 播客生成完成!")
    print(f"  总时长: {len(final_audio)/44100:.2f}s")
    print(f"  生成耗时: {total_time:.1f}s")
    print(f"  输出: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
