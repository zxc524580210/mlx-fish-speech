"""S2-Pro: optional reference-audio conditioning (same path as fish-speech generate_long)."""

import numpy as np
import mlx.core as mx
import torch

from mlx_fish_speech.conversation import Conversation, Message
from mlx_fish_speech.content_sequence import TextPart, VQPart

DEFAULT_REFERENCE_TRANSCRIPT = "你好，这是一段用于音色参考的语音。"


def reference_system_parts(ref_plain_text: str, prompt_codes: np.ndarray) -> list:
    t = ref_plain_text.strip()
    if not t.startswith("<|speaker:"):
        t = f"<|speaker:0|>{t}"
    return [
        TextPart(
            text="convert the provided text to speech reference to the following:\n\nText:\n"
        ),
        TextPart(text=t),
        TextPart(text="\n\nSpeech:\n"),
        VQPart(codes=torch.from_numpy(prompt_codes)),
    ]


def generate_with_reference(
    gen,
    tokenizer,
    text: str,
    prompt_codes: np.ndarray,
    reference_transcript: str,
    *,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_steps: int = 500,
) -> np.ndarray:
    conv = Conversation()
    conv.append(
        Message(
            role="system",
            parts=reference_system_parts(reference_transcript, prompt_codes),
            add_im_start=True,
            add_im_end=True,
        )
    )
    conv.append(
        Message(
            role="user",
            parts=[TextPart(text=text)],
            add_im_start=True,
            add_im_end=True,
        )
    )
    conv.append(
        Message(
            role="assistant",
            modality="voice",
            parts=[],
            add_im_start=True,
            add_im_end=False,
        )
    )

    encoded, _, _ = conv.encode_for_inference(
        tokenizer, num_codebooks=gen.config.num_codebooks
    )

    tokens = mx.array(encoded[0].numpy()[None, :])
    cb = mx.array(encoded[1:].numpy()[None, :, :])
    cache = None
    all_cb = []

    for step in range(max_steps):
        logits, _, hs, cache = gen.model(tokens, cache, codebook_tokens=cb)
        mx.eval(logits, hs, *[c for pair in cache for c in pair])
        l = logits[0, -1, :]
        if step < 10:
            mask = np.zeros(gen.config.vocab_size, dtype=np.float32)
            mask[gen.im_end_id] = -1e9
            l = l + mx.array(mask)
        tok = gen.sample_top_p(l, temperature, top_p, constrained=True)
        if tok == gen.im_end_id:
            break
        if gen.semantic_begin_id <= tok <= gen.semantic_end_id:
            cbs = gen.model.generate_codebooks(
                hidden_states=hs[:, -1:, :],
                semantic_token=tok,
                semantic_begin_id=gen.semantic_begin_id,
                temperature=temperature,
                top_p=top_p,
            )
            all_cb.append(cbs)
            tokens = mx.array([[tok]])
            cb = mx.array([[cbs]]).transpose(0, 2, 1)
        else:
            tokens = mx.array([[tok]])
            cb = None

    if not all_cb:
        raise ValueError("No audio frames generated")
    return np.array(all_cb, dtype=np.int32).T
