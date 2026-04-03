"""
PostModule Transformer for MLX DAC.

This module implements the 8-layer transformer that processes quantized
codes before upsampling. Critical for correct audio quality.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional


def precompute_freqs_cis(max_seq_len: int, head_dim: int, rope_base: float = 10000.0) -> mx.array:
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (rope_base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    # Return as (max_seq_len, head_dim/2, 2) with cos and sin
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    return mx.stack([cos, sin], axis=-1)


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x: (B, T, n_heads, head_dim)
        freqs_cis: (T, head_dim/2, 2) - cos and sin
    
    Returns:
        x with rotary embeddings applied
    """
    # Split x into pairs
    x_r = x.reshape(*x.shape[:-1], -1, 2)  # (B, T, n_heads, head_dim/2, 2)
    x0 = x_r[..., 0]  # (B, T, n_heads, head_dim/2)
    x1 = x_r[..., 1]  # (B, T, n_heads, head_dim/2)
    
    # Get cos and sin
    cos = freqs_cis[..., 0]  # (T, head_dim/2)
    sin = freqs_cis[..., 1]  # (T, head_dim/2)
    
    # Broadcast for batch and heads
    cos = cos[None, :, None, :]  # (1, T, 1, head_dim/2)
    sin = sin[None, :, None, :]  # (1, T, 1, head_dim/2)
    
    # Apply rotation
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    
    # Combine back
    y = mx.stack([y0, y1], axis=-1)
    return y.reshape(x.shape)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
    
    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.weight


class LayerScale(nn.Module):
    """Layer scale for transformer blocks."""
    
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_value
    
    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class FeedForward(nn.Module):
    """SwiGLU FeedForward network."""
    
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1_weight = mx.zeros((intermediate_size, dim))
        self.w2_weight = mx.zeros((dim, intermediate_size))
        self.w3_weight = mx.zeros((intermediate_size, dim))
    
    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        h1 = x @ self.w1_weight.T
        h3 = x @ self.w3_weight.T
        h = nn.silu(h1) * h3
        return h @ self.w2_weight.T


class Attention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Combined QKV projection
        total_head_dim = (n_heads + 2 * n_kv_heads) * head_dim
        self.wqkv_weight = mx.zeros((total_head_dim, dim))
        self.wo_weight = mx.zeros((dim, n_heads * head_dim))
    
    def __call__(
        self, 
        x: mx.array, 
        freqs_cis: mx.array, 
        mask: mx.array
    ) -> mx.array:
        B, T, _ = x.shape
        
        # Project to QKV
        qkv = x @ self.wqkv_weight.T
        
        # Split into Q, K, V
        # Q uses all heads, K/V use kv_heads (for GQA)
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size+kv_size]
        v = qkv[..., q_size+kv_size:]
        
        # Reshape for attention
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        
        # Transpose for attention: (B, n_heads, T, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Repeat KV heads if needed (GQA)
        if self.n_heads != self.n_kv_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        
        # Scaled dot-product attention
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        
        # Apply causal mask
        scores = mx.where(mask, scores, mx.array(float('-inf')))
        
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v  # (B, n_heads, T, head_dim)
        
        # Transpose back and project
        out = mx.transpose(out, (0, 2, 1, 3))  # (B, T, n_heads, head_dim)
        out = out.reshape(B, T, -1)
        
        return out @ self.wo_weight.T


class TransformerBlock(nn.Module):
    """Single transformer block with LayerScale."""
    
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        
        self.attention = Attention(dim, n_heads, n_kv_heads, head_dim)
        self.attention_norm = RMSNorm(dim)
        self.attention_layer_scale = LayerScale(dim)
        
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.ffn_norm = RMSNorm(dim)
        self.ffn_layer_scale = LayerScale(dim)
    
    def __call__(
        self, 
        x: mx.array, 
        freqs_cis: mx.array, 
        mask: mx.array
    ) -> mx.array:
        # Pre-norm attention with layer scale
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask)
        )
        # Pre-norm FFN with layer scale
        out = h + self.ffn_layer_scale(
            self.feed_forward(self.ffn_norm(h))
        )
        return out


class PostModule(nn.Module):
    """
    Post-quantization transformer for DAC RVQ.
    
    Processes quantized codes before upsampling to improve reconstruction.
    8 layers of transformer with RoPE and causal attention.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 3072,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        # Transformer layers
        self.layers = [
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, intermediate_size)
            for _ in range(n_layers)
        ]
        
        # Final norm
        self.norm = RMSNorm(dim)
        
        # Precomputed RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(max_seq_len, head_dim, rope_base)
        
        # Causal mask
        mask = mx.tril(mx.ones((max_seq_len, max_seq_len)))
        self.causal_mask = mask.astype(mx.bool_)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, C, T) - channels-first input from codebook lookup
        
        Returns:
            (B, C, T) - processed output
        """
        # Permute to (B, T, C)
        x = mx.transpose(x, (0, 2, 1))
        
        B, T, C = x.shape
        
        # Get RoPE frequencies and mask for this sequence length
        freqs_cis = self.freqs_cis[:T]
        mask = self.causal_mask[:T, :T]
        mask = mask[None, None, :, :]  # (1, 1, T, T)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        
        # Final norm
        x = self.norm(x)
        
        # Permute back to (B, C, T)
        x = mx.transpose(x, (0, 2, 1))
        
        return x
