"""
MLX Accelerated Text2Semantic Generation

This module provides fast semantic token generation using MLX,
compatible with Fish-Speech's ContentSequence and Conversation formats.

Supports both S1-mini and S2-Pro models.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

from mlx_fish_speech.tokenizer import FishTokenizer, IM_END_TOKEN, MODALITY_TOKENS
from mlx_fish_speech.content_sequence import ContentSequence, TextPart, VQPart
from mlx_fish_speech.conversation import Conversation, Message

from mlx_fish_speech.models.text2semantic import DualARTransformer


def _resolve_tokenizer_path(weights_path: Path) -> str:
    """Find the tokenizer path for the given model weights.
    
    All Fish-Speech models (S1-mini, S2-Pro) share the same base tokenizer
    (Qwen-based, im_end=151645). Only semantic token ranges differ, and those
    come from model config, not the tokenizer. So S2-Pro's tokenizer.json is
    compatible with all models.
    
    Checks in order:
    1. The weights dir itself (if it has tokenizer.json)
    2. Parent checkpoints directory with matching name
    """
    def _has_hf_tokenizer(p: Path) -> bool:
        return (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
    
    candidates = [
        weights_path,
        weights_path.parent / "s2-pro",  # fallback to s2-pro in same checkpoints dir
    ]
    
    for path in candidates:
        if path.exists() and _has_hf_tokenizer(path):
            return str(path)
    
    raise FileNotFoundError(
        f"Could not find tokenizer for {weights_path}. "
        f"Ensure tokenizer files exist in the model directory."
    )


class MLXGenerator:
    """MLX-based Text2Semantic generator compatible with Fish-Speech.
    
    Supports both S1-mini (flat config) and S2-Pro (fish_qwen3_omni) models.
    """
    
    def __init__(self, weights_path: str):
        print("[MLX] Loading Text2Semantic model...")
        self.weights_path = Path(weights_path)
        self.model = DualARTransformer.from_pretrained(weights_path)
        self.config = self.model.config
        
        # Load Fish-Speech tokenizer
        tokenizer_path = _resolve_tokenizer_path(self.weights_path)
        print(f"[MLX] Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = FishTokenizer.from_pretrained(tokenizer_path)
        
        # Key token IDs - prefer config values (>= 0), fallback to tokenizer
        self.semantic_begin_id = self.config.semantic_begin_id
        self.semantic_end_id = self.config.semantic_end_id
        
        def _token_id(config_val: int, token_str: str) -> int:
            return config_val if config_val >= 0 else self.tokenizer.get_token_id(token_str)
        
        self.im_start_id = _token_id(self.config.im_start_id, "<|im_start|>")
        self.im_end_id = _token_id(self.config.im_end_id, IM_END_TOKEN)
        self.pad_id = _token_id(self.config.pad_id, "<|pad|>")
        self.voice_id = _token_id(self.config.voice_id, "<|voice|>")
        self.interleave_id = self.tokenizer.get_token_id("<|interleave|>")
        
        # Detect prompt format based on model type
        # S1-mini uses ContentSequence(interleave) format where <|interleave|> is trained
        # S2-Pro uses Conversation chat template where <|im_start|> is trained
        self.use_interleave_format = self._detect_prompt_format()
        
        # Build semantic logit bias for constrained decoding
        self._semantic_logit_bias = self._build_semantic_logit_bias()
        
        fmt = "interleave" if self.use_interleave_format else "chat_template"
        print(f"[MLX] Model loaded: {self.config.n_layer} layers, dim={self.config.dim}")
        print(f"[MLX] semantic range: [{self.semantic_begin_id}, {self.semantic_end_id}]")
        print(f"[MLX] im_end_id: {self.im_end_id}, prompt_format: {fmt}")
    
    def _build_semantic_logit_bias(self) -> mx.array:
        """Build bias tensor that constrains decoding to only semantic tokens + im_end."""
        # Build as numpy first to avoid -inf + inf = NaN
        import numpy as np
        bias = np.full((self.config.vocab_size,), -np.inf, dtype=np.float32)
        bias[self.semantic_begin_id:self.semantic_end_id + 1] = 0.0
        bias[self.im_end_id] = 0.0
        return mx.array(bias)
    
    def _detect_prompt_format(self) -> bool:
        """Detect whether to use ContentSequence interleave format.
        
        S1-mini: <|interleave|> is trained, <|im_start|>/<|voice|> are NOT trained.
                 Uses ContentSequence(modality='interleave') format.
        S2-Pro:  <|im_start|>/<|voice|> are trained.
                 Uses Conversation chat template format.
        
        Heuristic: S1-mini has n_layer=28, dim=1024 (smaller model).
        S2-Pro has n_layer=32+, dim=2048+ (larger model).
        """
        # S1-mini: 28 layers, 1024 dim
        if self.config.n_layer <= 28 and self.config.dim <= 1024:
            return True
        return False
    
    def encode_text(self, text: str) -> mx.array:
        """Encode text into prompt tokens.
        
        Auto-selects format based on model type:
        - S1-mini: ContentSequence interleave format
          <|interleave|><|speaker:0|>{text}
        - S2-Pro: Chat template format  
          <|im_start|>system\n...\n<|im_start|>user\n{text}\n<|im_start|>assistant\n<|voice|>
        """
        if self.use_interleave_format:
            return self._encode_text_interleave(text)
        else:
            return self._encode_text_chat_template(text)
    
    def _encode_text_interleave(self, text: str) -> mx.array:
        """Encode text using ContentSequence interleave format (S1-mini).
        
        This matches the original fish-speech generate_long() which uses:
            ContentSequence(modality='interleave')
            cs.append([TextPart(text=text)], add_end=False, speaker=0)
        
        Produces: <|interleave|><|speaker:0|>{text}
        """
        # <|speaker:0|> is encoded as regular text, not a special token
        speaker_tokens = self.tokenizer.encode(
            "<|speaker:0|>",
            add_special_tokens=False,
        )
        text_tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        
        prompt = [self.interleave_id] + speaker_tokens + text_tokens
        return mx.array(np.array(prompt, dtype=np.int32))
    
    def _encode_text_chat_template(self, text: str) -> mx.array:
        """Encode text using chat template format (S2-Pro).
        
        Template:
            <|im_start|>system\nconvert the provided text to speech<|im_end|>\n
            <|im_start|>user\n{text}<|im_end|>\n
            <|im_start|>assistant\n<|voice|>
        """
        text_tokens = self.tokenizer.encode(
            "system\nconvert the provided text to speech",
            add_special_tokens=False,
        )
        user_tokens = self.tokenizer.encode(
            "user\n" + text,
            add_special_tokens=False,
        )
        assistant_tokens = self.tokenizer.encode(
            "assistant\n",
            add_special_tokens=False,
        )
        
        prompt = (
            [self.im_start_id] + text_tokens + [self.im_end_id, 198]  # 198 = \n
            + [self.im_start_id] + user_tokens + [self.im_end_id, 198]
            + [self.im_start_id] + assistant_tokens + [self.voice_id]
        )
        
        return mx.array(np.array(prompt, dtype=np.int32))
    
    def sample_top_p(self, logits: mx.array, temperature: float, top_p: float,
                     constrained: bool = True) -> int:
        """Sample from logits using temperature and top-p sampling.
        
        Args:
            logits: Raw logits from the model
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            constrained: If True, apply semantic logit bias
        """
        # Apply semantic constraint
        if constrained:
            logits = logits + self._semantic_logit_bias
        
        # Compute probs to determine top-p cutoff
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        
        # Keep tokens until cumsum > top_p (always keep at least one)
        sorted_mask = cumsum > top_p
        sorted_mask = mx.concatenate([mx.zeros((1,), dtype=mx.bool_), sorted_mask[:-1]], axis=-1)
        
        # Mask out low probability tokens in logit space
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_logits = mx.where(sorted_mask, mx.array(-float('inf')), sorted_logits)
        
        # Apply temperature
        sorted_logits = sorted_logits / max(temperature, 1e-5)
        
        # Sample
        final_probs = mx.softmax(sorted_logits, axis=-1)
        idx = mx.random.categorical(mx.log(final_probs + 1e-10))
        
        return int(sorted_indices[idx.item()])
    
    def generate(
        self,
        text: str,
        max_new_tokens: int = 1024,
        min_new_tokens: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
    ) -> np.ndarray:
        """
        Generate multi-codebook codes from text.
        
        Returns:
            codes: np.ndarray of shape (num_codebooks, time)
        """
        
        # Encode text
        prompt_tokens = self.encode_text(text)
        prompt_len = len(prompt_tokens)
        
        print(f"[MLX] Prompt tokens: {prompt_len}")
        
        mx.set_cache_limit(4 * 1024 * 1024 * 1024)
        
        # Initialize
        tokens = prompt_tokens[None, :]  # (1, L)
        cache = None
        
        all_semantic_tokens = []
        all_codebook_tokens = []
        
        start_time = time.time()
        n_generated = 0
        codebook_tokens = None
        
        # Pre-compute im_end mask (reuse across steps)
        im_end_mask_arr = None
        
        for step in range(max_new_tokens):
            
            # ── GPU slow decoder forward pass ──
            token_logits, codebook_logits, hidden_states, cache = self.model(
                tokens, cache, codebook_tokens=codebook_tokens
            )
            
            # Single eval: logits + hidden + cache
            mx.eval(token_logits, hidden_states, *[c for pair in cache for c in pair])

            
            if (step + 1) % 500 == 0:
                mx.clear_cache()
            
            # ── Sample next token ──
            logits = token_logits[0, -1, :]
            
            # Block im_end before min_new_tokens to prevent premature stopping
            if step < min_new_tokens:
                if im_end_mask_arr is None:
                    im_end_mask_np = np.zeros(self.config.vocab_size, dtype=np.float32)
                    im_end_mask_np[self.im_end_id] = -1e9
                    im_end_mask_arr = mx.array(im_end_mask_np)
                logits = logits + im_end_mask_arr
            
            next_token = self.sample_top_p(logits, temperature, top_p, constrained=True)
            n_generated += 1
            
            if next_token == self.im_end_id:
                print(f"[MLX] IM_END at step {step}")
                break
            
            all_semantic_tokens.append(next_token)
            
            # ── Fast decoder (codebook generation) ──
            if self.semantic_begin_id <= next_token <= self.semantic_end_id:
                last_hidden = hidden_states[:, -1:, :]
                
                # GPU fast decoder
                codebooks = self.model.generate_codebooks(
                    hidden_states=last_hidden,
                    semantic_token=next_token,
                    semantic_begin_id=self.semantic_begin_id,
                    temperature=temperature,
                    top_p=top_p,
                )
                all_codebook_tokens.append(codebooks)
                tokens = mx.array([[next_token]])
                codebook_tokens = mx.array([[codebooks]]).transpose(0, 2, 1)
            else:
                tokens = mx.array([[next_token]])
                codebook_tokens = None
            
            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[MLX] Step {step+1}: {n_generated/elapsed:.1f} tok/s")
        
        elapsed = time.time() - start_time
        n_frames = len(all_codebook_tokens)
        
        print(f"[MLX] Generated {n_generated} tokens, {n_frames} audio frames in {elapsed:.2f}s")
        print(f"[MLX] Speed: {n_generated/elapsed:.1f} tok/s")
        
        if n_frames == 0:
            raise ValueError("No audio tokens generated. Model may need different prompting.")
        
        codes = np.array(all_codebook_tokens, dtype=np.int32).T
        return codes


def main():
    """CLI: run MLXGenerator on a short prompt (writes codes to outputs/)."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, default="你好，欢迎使用语音合成。")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model path. Default: auto-detect s2-pro or s1-mini")
    args = parser.parse_args()
    
    # Auto-detect model
    if args.model:
        weights_path = Path(args.model)
    else:
        # Try common locations
        search_paths = [
            PROJECT_DIR / "weights" / "s2-pro",
            PROJECT_DIR / "weights" / "openaudio-s1-mini",
            PROJECT_DIR / "checkpoints" / "s2-pro",
        ]
        weights_path = None
        for p in search_paths:
            if p.exists():
                weights_path = p
                break
        if weights_path is None:
            raise FileNotFoundError("No model found. Please specify --model path.")
    
    generator = MLXGenerator(str(weights_path))
    
    print(f"\n[MLX] Generating for: \"{args.text}\"")
    codes = generator.generate(args.text, max_new_tokens=args.max_tokens)
    print(f"[MLX] Output codes shape: {codes.shape}")
    
    # Save codes
    output_path = PROJECT_DIR / "outputs/mlx_codes.npy"
    output_path.parent.mkdir(exist_ok=True)
    np.save(output_path, codes)
    print(f"[MLX] Saved to: {output_path}")


if __name__ == "__main__":
    main()
