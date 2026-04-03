"""
DualAR Transformer for Text-to-Semantic token generation.

This module implements the core language model that converts text tokens
to semantic audio tokens using a dual autoregressive architecture.

Supports both S1-mini (flat config) and S2-Pro (fish_qwen3_omni nested config).
"""

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DualARConfig:
    """Configuration for DualARTransformer.
    
    Supports both S1-mini (flat config) and S2-Pro (fish_qwen3_omni nested config).
    """
    
    model_type: str = "dual_ar"
    
    vocab_size: int = 155776
    n_layer: int = 28
    n_head: int = 16
    dim: int = 1024
    intermediate_size: int = 3072
    n_local_heads: int = 8
    head_dim: int = 128
    rope_base: float = 1000000
    norm_eps: float = 1e-6
    max_seq_len: int = 8192
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = True
    
    # Codebook settings
    codebook_size: int = 4096
    num_codebooks: int = 10
    
    # Semantic token settings (for codebook embedding addition)
    semantic_begin_id: int = 151658  # Default for Fish-Speech S1-mini
    semantic_end_id: int = 155753    # semantic_begin_id + codebook_size - 1
    scale_codebook_embeddings: bool = True
    norm_fastlayer_input: bool = False
    
    # Special token IDs (model-specific, decoupled from tokenizer)
    # -1 = use tokenizer lookup (fallback); explicit values preferred
    im_start_id: int = -1     # <|im_start|> token ID
    im_end_id: int = -1       # <|im_end|> token ID
    pad_id: int = -1          # <|pad|> token ID
    voice_id: int = -1        # <|voice|> token ID
    
    # Fast decoder settings
    n_fast_layer: int = 4
    fast_dim: int = 1024
    fast_n_head: int = 16
    fast_n_local_heads: int = 8
    fast_head_dim: int = 64
    fast_intermediate_size: int = 3072
    fast_attention_qkv_bias: bool = False
    fast_attention_qk_norm: bool = False
    fast_attention_o_bias: bool = False
    
    def __post_init__(self):
        """Fill in defaults for fast decoder fields that mirror slow decoder."""
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = n_hidden + (256 - n_hidden % 256) % 256
        if self.fast_dim is None:
            self.fast_dim = self.dim
        if self.fast_n_head is None:
            self.fast_n_head = self.n_head
        if self.fast_n_local_heads is None:
            self.fast_n_local_heads = self.n_local_heads
        if self.fast_head_dim is None:
            self.fast_head_dim = self.head_dim
        if self.fast_intermediate_size is None:
            self.fast_intermediate_size = self.intermediate_size
    
    @classmethod
    def from_pretrained(cls, path: str) -> "DualARConfig":
        """Load config from pretrained model directory."""
        config_path = Path(path) / "config.json"
        with open(config_path) as f:
            data = json.load(f)
        
        model_type = data.get("model_type", "dual_ar")
        
        if model_type == "fish_qwen3_omni":
            return cls._from_fish_qwen3_omni(data)
        
        # Flat config (S1-mini / dual_ar)
        valid_keys = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**kwargs)
    
    @staticmethod
    def _from_fish_qwen3_omni(data: dict) -> "DualARConfig":
        """Parse nested fish_qwen3_omni config (S2-Pro format)."""
        tc = data["text_config"]
        adc = data["audio_decoder_config"]
        flat = dict(
            model_type="dual_ar",
            vocab_size=tc["vocab_size"],
            n_layer=tc["n_layer"],
            n_head=tc["n_head"],
            n_local_heads=tc.get("n_local_heads", -1),
            head_dim=tc.get("head_dim"),
            dim=tc["dim"],
            intermediate_size=tc.get("intermediate_size"),
            rope_base=tc.get("rope_base", 10000),
            norm_eps=tc.get("norm_eps", 1e-5),
            max_seq_len=tc.get("max_seq_len", 2048),
            tie_word_embeddings=tc.get("tie_word_embeddings", True),
            attention_qkv_bias=tc.get("attention_qkv_bias", False),
            attention_o_bias=tc.get("attention_o_bias", False),
            attention_qk_norm=tc.get("attention_qk_norm", False),
            semantic_begin_id=data.get("semantic_start_token_id", 0),
            semantic_end_id=data.get("semantic_end_token_id", 0),
            scale_codebook_embeddings=True,
            norm_fastlayer_input=True,
            codebook_size=adc["vocab_size"],
            num_codebooks=adc["num_codebooks"],
            n_fast_layer=adc["n_layer"],
            fast_dim=adc.get("dim"),
            fast_n_head=adc.get("n_head"),
            fast_n_local_heads=adc.get("n_local_heads"),
            fast_head_dim=adc.get("head_dim"),
            fast_intermediate_size=adc.get("intermediate_size"),
            fast_attention_qkv_bias=adc.get("attention_qkv_bias", False),
            fast_attention_qk_norm=adc.get("attention_qk_norm", False),
            fast_attention_o_bias=adc.get("attention_o_bias", False),
        )
        valid_keys = {f.name for f in fields(DualARConfig)}
        flat = {k: v for k, v in flat.items() if k in valid_keys and v is not None}
        return DualARConfig(**flat)


class AudioProjector(nn.Module):
    """
    Audio projector with weight keys matching Fish-Speech structure.
    Original uses indices 0 and 2 for Linear layers, we remap to linear1/linear2.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.silu(x)
        x = self.linear2(x)
        return x



class Attention(nn.Module):
    """Multi-head attention with fused QKV projection and optional QK normalization."""
    
    def __init__(self, dim: int, n_head: int, n_kv_head: int, head_dim: int, 
                 norm_eps: float = 1e-6, use_qk_norm: bool = True,
                 qkv_bias: bool = False, o_bias: bool = False,
                 rope_base: float = 1000000.0):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        
        # Fused QKV projection
        self.wqkv = nn.Linear(dim, (n_head + 2 * n_kv_head) * head_dim, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=o_bias)
        
        # Fused RoPE (uses mx.fast.rope internally)
        self.rotary_emb = nn.RoPE(head_dim, traditional=True, base=rope_base)
        
        # QK normalization
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape
        
        qkv = self.wqkv(x)
        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_head * self.head_dim
        
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)
        
        q = q.reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        
        # QK normalization before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Fused RoPE with offset from cache
        if cache is not None:
            q = self.rotary_emb(q, offset=cache[0].shape[2])
            k = self.rotary_emb(k, offset=cache[0].shape[2])
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        else:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        
        new_cache = (k, v)
        
        # Fused scaled dot-product attention (handles GQA internally)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        ).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        return self.wo(out), new_cache



class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, dim: int, n_head: int, n_kv_head: int, head_dim: int, 
                 intermediate_size: int, norm_eps: float, use_qk_norm: bool = True,
                 qkv_bias: bool = False, o_bias: bool = False,
                 rope_base: float = 1000000.0):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, head_dim, norm_eps, use_qk_norm,
            qkv_bias=qkv_bias, o_bias=o_bias, rope_base=rope_base
        )
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Self-attention with residual
        attn_out, new_cache = self.attention(
            self.attention_norm(x), mask, cache
        )
        x = x + attn_out
        
        # FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x, new_cache


class DualARTransformer(nn.Module):
    """
    Dual Autoregressive Transformer for Text-to-Semantic generation.
    
    Weight key structure (matching Fish-Speech):
    - embeddings.weight: Text token embeddings
    - codebook_embeddings.weight: Audio codebook embeddings  
    - fast_embeddings.weight: Fast decoder embeddings
    - layers.N.attention.wqkv/wo: Slow decoder attention
    - layers.N.feed_forward.w1/w2/w3: Slow decoder FFN
    - fast_layers.N.*: Fast decoder layers
    - output.weight: LM head
    """
    
    def __init__(self, config: DualARConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (matching weight keys)
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        
        # Project from slow dim to fast dim if they differ
        if config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim, bias=False)
        else:
            self.fast_project_in = None
        
        # Slow decoder (main transformer)
        self.layers = [
            TransformerBlock(
                config.dim, config.n_head, config.n_local_heads, 
                config.head_dim, config.intermediate_size, config.norm_eps,
                use_qk_norm=config.attention_qk_norm,
                qkv_bias=config.attention_qkv_bias,
                o_bias=config.attention_o_bias,
                rope_base=config.rope_base,
            ) 
            for _ in range(config.n_layer)
        ]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        
        # Fast decoder layers
        self.fast_layers = [
            TransformerBlock(
                config.fast_dim, config.fast_n_head, config.fast_n_local_heads,
                config.fast_head_dim, config.fast_intermediate_size, config.norm_eps,
                use_qk_norm=config.fast_attention_qk_norm,
                qkv_bias=config.fast_attention_qkv_bias,
                o_bias=config.fast_attention_o_bias,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = nn.RMSNorm(config.fast_dim, eps=config.norm_eps)
        
        # Output heads
        if not config.tie_word_embeddings:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        else:
            self.output = None  # Will use embeddings.weight
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        
        # Pre-compute codebook embedding offsets for vectorized lookup
        self._cb_offsets = mx.arange(config.num_codebooks).reshape(1, config.num_codebooks, 1) * config.codebook_size
    
    def _compute_logits(self, x: mx.array) -> mx.array:
        """Compute token logits, handling tied embeddings."""
        if self.config.tie_word_embeddings:
            return x @ self.embeddings.weight.T
        else:
            return self.output(x)
    
    def __call__(
        self,
        tokens: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        codebook_tokens: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, List[Tuple[mx.array, mx.array]]]:
        """
        Forward pass with optional multi-codebook input.
        
        Args:
            tokens: Input token IDs [B, L]
            cache: Optional KV cache (list of (K, V) tuples)
            codebook_tokens: Optional codebook indices [B, num_codebooks, L]
            
        Returns:
            token_logits, None, hidden_states, new_cache
        """
        B, L = tokens.shape
        
        # Get position offset from cache for mask padding
        offset = 0 if cache is None else cache[0][0].shape[2]
        
        # Embed tokens
        x = self.embeddings(tokens)
        
        # Add codebook embeddings if provided (vectorized: single gather + sum)
        if codebook_tokens is not None:
            # Vectorized: [B, num_codebooks, L] + offsets → single embedding lookup
            all_indices = (codebook_tokens + self._cb_offsets).reshape(B, -1)  # [B, num_codebooks*L]
            all_embeds = self.codebook_embeddings(all_indices)  # [B, num_codebooks*L, dim]
            all_embeds = all_embeds.reshape(B, self.config.num_codebooks, L, -1)
            vq_embeds_sum = all_embeds.sum(axis=1)  # [B, L, dim]
            
            is_semantic = (tokens >= self.config.semantic_begin_id) & (tokens <= self.config.semantic_end_id)
            is_semantic = mx.expand_dims(is_semantic, axis=-1)
            vq_embeds_sum = mx.where(is_semantic, vq_embeds_sum, mx.zeros_like(vq_embeds_sum))
            x = x + vq_embeds_sum
            
            if self.config.scale_codebook_embeddings:
                scale = 1.0 / math.sqrt(self.config.num_codebooks + 1)
                x = mx.where(is_semantic, x * scale, x)
        
        # Create causal mask (cast to match model dtype)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
        if offset > 0:
            mask = mx.pad(mask, [(0, 0), (offset, 0)])
        
        # Slow decoder (RoPE offset handled automatically by nn.RoPE + cache)
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x, c = layer(x, mask, layer_cache)
            new_cache.append(c)
        
        slow_out = self.norm(x)
        hidden_states = slow_out if self.config.norm_fastlayer_input else x
        token_logits = self._compute_logits(slow_out)
        
        return token_logits, None, hidden_states, new_cache
    
    def generate_codebooks(
        self, 
        hidden_states: mx.array,  # (B, 1, dim) - slow decoder hidden states
        semantic_token: int,  # The sampled semantic token from slow decoder
        semantic_begin_id: int,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> List[int]:
        """
        Generate codebooks 0-9 using iterative fast decoder with KV cache.
        
        Zero-sync version: all operations stay on GPU as lazy computation.
        Only ONE mx.eval() at the end materializes all codebook tokens.
        This eliminates ~9 CPU↔GPU sync barriers (~72ms saved).
        """
        # CB0: derive from semantic token (Python int, no GPU needed)
        cb0 = semantic_token - semantic_begin_id
        if cb0 < 0:
            cb0 = 0
        if cb0 >= self.config.codebook_size:
            cb0 = self.config.codebook_size - 1
        
        # Collect codebook tokens as mx.arrays (stay on GPU)
        cb_tokens_mx = [mx.array(cb0)]
        
        # Initialize KV cache for fast layers
        fast_cache = [None] * len(self.fast_layers)
        
        # Step 1: Prime fast decoder (NO eval — stays lazy)
        fast_x = hidden_states
        
        if self.fast_project_in is not None:
            fast_x = self.fast_project_in(fast_x)
        
        for i, layer in enumerate(self.fast_layers):
            fast_x, fast_cache[i] = layer(fast_x, None, fast_cache[i])
        
        # NO mx.eval() here — keep lazy!
        
        # Step 2: Embed CB0 and generate codebooks 1-9
        prev_embed = self.fast_embeddings(mx.array([[cb0]]))
        cache_len = 1  # Track manually instead of reading .shape[2]
        
        for cb_idx in range(1, self.config.num_codebooks):
            fast_x = prev_embed
            
            mask = nn.MultiHeadAttention.create_additive_causal_mask(1).astype(fast_x.dtype)
            mask = mx.pad(mask, [(0, 0), (cache_len, 0)])
            
            for i, layer in enumerate(self.fast_layers):
                fast_x, fast_cache[i] = layer(fast_x, mask, fast_cache[i])
            
            cache_len += 1
            
            # Get logits and sample — ALL on GPU, no .item()!
            fast_x_norm = self.fast_norm(fast_x)
            logits = self.fast_output(fast_x_norm)[0, -1, :]
            
            logits = logits / temperature
            probs = mx.softmax(logits, axis=-1)
            
            sorted_indices = mx.argsort(-probs)
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumsum = mx.cumsum(sorted_probs)
            mask_p = cumsum <= top_p
            mask_p = mx.concatenate([mx.ones((1,)), mask_p[:-1]])
            sorted_probs = sorted_probs * mask_p
            sorted_probs = sorted_probs / (mx.sum(sorted_probs) + 1e-10)
            
            idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))
            cb_token_mx = sorted_indices[idx]  # Stay as mx.array!
            cb_token_mx = mx.minimum(cb_token_mx, self.config.codebook_size - 1)
            
            cb_tokens_mx.append(cb_token_mx)
            
            # Embed for next iteration using mx.array index (no .item()!)
            prev_embed = self.fast_embeddings(cb_token_mx.reshape(1, 1))
        
        # Single eval at the end — materialize all codebook tokens at once
        mx.eval(*cb_tokens_mx)
        codebooks = [int(t.item()) for t in cb_tokens_mx]
        
        return codebooks
    
    def generate_codebooks_batch(
        self,
        hidden_states: mx.array,  # (B, 1, dim)
        semantic_tokens: mx.array,  # (B,) semantic token IDs
        semantic_begin_id: int,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> mx.array:
        """
        Batched codebook generation — processes all B sequences simultaneously.
        
        Returns: mx.array of shape (B, num_codebooks) with codebook token IDs.
        All operations stay on GPU as lazy computation with single mx.eval at end.
        """
        B = hidden_states.shape[0]
        
        # CB0: derive from semantic tokens (batched)
        cb0 = semantic_tokens - semantic_begin_id
        cb0 = mx.clip(cb0, 0, self.config.codebook_size - 1)
        
        # Collect all codebook tokens: list of [B] arrays
        all_cb = [cb0]
        
        # Initialize KV cache for fast layers (batched)
        fast_cache = [None] * len(self.fast_layers)
        
        # Step 1: Prime fast decoder with hidden states [B, 1, dim]
        fast_freqs_cis = self._fast_freqs_cis[:1]
        fast_x = hidden_states
        
        if self.fast_project_in is not None:
            fast_x = self.fast_project_in(fast_x)
        
        for i, layer in enumerate(self.fast_layers):
            fast_x, fast_cache[i] = layer(fast_x, fast_freqs_cis, None, fast_cache[i])
        
        # Step 2: Embed CB0 and generate codebooks 1-9
        prev_embed = self.fast_embeddings(cb0.reshape(B, 1))  # [B, 1, fast_dim]
        cache_len = 1
        
        for cb_idx in range(1, self.config.num_codebooks):
            fast_freqs_cis = self._fast_freqs_cis[cb_idx:cb_idx+1]
            fast_x = prev_embed
            
            mask = nn.MultiHeadAttention.create_additive_causal_mask(1)
            mask = mx.pad(mask, [(0, 0), (cache_len, 0)])
            
            for i, layer in enumerate(self.fast_layers):
                fast_x, fast_cache[i] = layer(fast_x, fast_freqs_cis, mask, fast_cache[i])
            
            cache_len += 1
            
            # Batched logits and sampling
            fast_x_norm = self.fast_norm(fast_x)
            logits = self.fast_output(fast_x_norm)[:, -1, :]  # [B, codebook_size]
            
            logits = logits / temperature
            probs = mx.softmax(logits, axis=-1)
            
            # Batched top-p sampling
            sorted_indices = mx.argsort(-probs, axis=-1)  # [B, codebook_size]
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            mask_p = cumsum <= top_p
            mask_p = mx.concatenate([mx.ones((B, 1)), mask_p[:, :-1]], axis=1)
            sorted_probs = sorted_probs * mask_p
            sorted_probs = sorted_probs / (mx.sum(sorted_probs, axis=-1, keepdims=True) + 1e-10)
            
            # Sample from filtered distribution
            idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))  # [B]
            cb_token = mx.take_along_axis(sorted_indices, idx.reshape(B, 1), axis=1).squeeze(1)  # [B]
            cb_token = mx.minimum(cb_token, self.config.codebook_size - 1)
            
            all_cb.append(cb_token)
            
            # Embed for next iteration [B, 1, fast_dim]
            prev_embed = self.fast_embeddings(cb_token.reshape(B, 1))
        
        # Stack and materialize
        result = mx.stack(all_cb, axis=1)  # [B, num_codebooks]
        mx.eval(result)
        return result
    
    @classmethod
    def from_pretrained(cls, path: str) -> "DualARTransformer":
        """Load model from pretrained weights.
        
        Supports:
        - MLX safetensors (text2semantic.safetensors)
        - PyTorch model.pth
        - Sharded safetensors (model.safetensors.index.json, for S2-Pro)
        """
        path = Path(path)
        
        # Load config
        config = DualARConfig.from_pretrained(path)
        print(f"Loading DualARTransformer: dim={config.dim}, n_layer={config.n_layer}, "
              f"fast_dim={config.fast_dim}, norm_fastlayer_input={config.norm_fastlayer_input}")
        
        # Create model
        model = cls(config)
        
        # Try loading in priority order
        mlx_weights_path = path / "text2semantic.safetensors"
        index_json = path / "model.safetensors.index.json"
        single_st = path / "model.safetensors"
        pth_file = path / "model.pth"
        
        if mlx_weights_path.exists():
            # Pre-converted MLX weights
            weights = dict(mx.load(str(mlx_weights_path)))
        elif index_json.exists():
            # Sharded safetensors (S2-Pro)
            print(f"Loading sharded safetensors...")
            with open(index_json) as f:
                st_index = json.load(f)
            shard_files = sorted(set(st_index["weight_map"].values()))
            weights = {}
            for shard in shard_files:
                print(f"  Loading shard: {shard}")
                shard_weights = mx.load(str(path / shard))
                weights.update(shard_weights)
            weights = _remap_fish_qwen3_omni_keys(weights)
        elif (shards := sorted(path.glob("model-*-of-*.safetensors"))) and len(shards) >= 1:
            # Same shards as HuggingFace layout but index.json missing (incomplete copy)
            print("Loading sharded safetensors (no index.json, merging by shard name)...")
            weights = {}
            for shard in shards:
                print(f"  Loading shard: {shard.name}")
                weights.update(dict(mx.load(str(shard))))
            weights = _remap_fish_qwen3_omni_keys(weights)
        elif single_st.exists():
            # Single safetensors
            weights = dict(mx.load(str(single_st)))
            weights = _remap_fish_qwen3_omni_keys(weights)
        elif pth_file.exists():
            # PyTorch weights (need conversion)
            import torch
            import numpy as np
            pt_weights = torch.load(str(pth_file), map_location="cpu", weights_only=True)
            if "state_dict" in pt_weights:
                pt_weights = pt_weights["state_dict"]
            weights = {}
            for k, v in pt_weights.items():
                if k.startswith("model."):
                    k = k[6:]
                weights[k] = mx.array(v.float().numpy())
        else:
            raise FileNotFoundError(f"No model weights found in {path}")
        
        # Remap audio_projector keys
        remapped = {}
        for key, value in weights.items():
            if key.startswith("audio_projector.0."):
                new_key = key.replace("audio_projector.0.", "audio_projector.linear1.")
                remapped[new_key] = value
            elif key.startswith("audio_projector.2."):
                new_key = key.replace("audio_projector.2.", "audio_projector.linear2.")
                remapped[new_key] = value
            else:
                remapped[key] = value
        
        # Load weights, allowing strict=False behavior
        model.load_weights(list(remapped.items()), strict=False)
        print(f"Loaded {len(remapped)} weight tensors")
        
        return model


def _remap_fish_qwen3_omni_keys(weights: dict) -> dict:
    """Remap fish_qwen3_omni weight keys to DualARTransformer format.
    
    Maps:
    - text_model.model.* -> * (remove prefix)
    - audio_decoder.* -> fast_* (add prefix)
    - audio_decoder.codebook_embeddings.* -> codebook_embeddings.* (keep as-is)
    """
    if not any(k.startswith(("text_model.", "audio_decoder.")) for k in weights):
        return weights
    
    new_weights = {}
    for k, v in weights.items():
        if k.startswith("text_model.model."):
            new_key = k[len("text_model.model."):]
        elif k.startswith("audio_decoder."):
            suffix = k[len("audio_decoder."):]
            new_key = (
                suffix
                if suffix.startswith("codebook_embeddings.")
                else "fast_" + suffix
            )
        else:
            new_key = k
        new_weights[new_key] = v
    return new_weights
