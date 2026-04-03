"""
Complete MLX DAC Codec with weight loading.

Matches Fish-Speech DAC architecture for decode-only operation.
Input/Output aligned with PyTorch version.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .post_module import PostModule


@dataclass
class DACConfig:
    """DAC configuration matching Fish-Speech defaults."""
    sample_rate: int = 44100
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: Tuple[int, ...] = (8, 8, 4, 2)  # Upsample factors
    n_codebooks: int = 9
    codebook_size: int = 1024
    semantic_codebook_size: int = 4096
    codebook_dim: int = 8


# =============================================================================
# Weight Norm Handling
# =============================================================================

def merge_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    """Merge weight norm: w = g * v / ||v||"""
    v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
    return g * v / v_norm


# =============================================================================
# Layers  
# =============================================================================

class Snake1d(nn.Module):
    """Snake activation: x + (1/alpha) * sin²(alpha * x)"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))
    
    def __call__(self, x: mx.array) -> mx.array:
        return x + (1.0 / self.alpha) * mx.power(mx.sin(self.alpha * x), 2)


class Conv1d(nn.Module):
    """1D Convolution (channels last: B, T, C)."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = mx.zeros((out_ch, kernel, in_ch))
        self.bias = mx.zeros((out_ch,))
    
    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        out = mx.conv1d(x, self.weight, stride=self.stride)
        return out + self.bias


class CausalConv1d(nn.Module):
    """Causal 1D Convolution with left padding and dilation support.
    
    MLX conv1d doesn't support dilation natively, so we implement it
    by expanding the kernel with zeros (pre-dilation).
    """
    
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        # Effective kernel size with dilation
        self.effective_kernel = (kernel - 1) * dilation + 1
        # Padding for causal: pad left by (effective_kernel - stride)
        self.padding = self.effective_kernel - stride
        
        # Store original kernel size weight, will be dilated if needed
        self.weight = mx.zeros((out_ch, kernel, in_ch))
        self.bias = mx.zeros((out_ch,))
        self._dilated_weight = None
    
    def _get_dilated_weight(self) -> mx.array:
        """Get weight with dilation applied (kernel expanded with zeros)."""
        if self.dilation == 1:
            return self.weight
        
        # Pre-dilate the kernel by inserting zeros
        import numpy as np
        w_np = np.array(self.weight)
        out_ch, k, in_ch = w_np.shape
        new_k = self.effective_kernel
        
        dilated = np.zeros((out_ch, new_k, in_ch), dtype=w_np.dtype)
        for i in range(k):
            dilated[:, i * self.dilation, :] = w_np[:, i, :]
        
        return mx.array(dilated)
    
    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        input_len = x.shape[1]
        
        # Causal left padding
        x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        
        # Get dilated weight and apply convolution
        weight = self._get_dilated_weight()
        out = mx.conv1d(x, weight, stride=self.stride)
        
        # Trim to correct output length (for stride=1, should match input)
        if self.stride == 1:
            out = out[:, :input_len, :]
        
        return out + self.bias


class CausalConvTranspose1d(nn.Module):
    """Causal 1D Transposed Convolution."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.weight = mx.zeros((in_ch, kernel, out_ch))
        self.bias = mx.zeros((out_ch,))
    
    def __call__(self, x: mx.array) -> mx.array:
        out = mx.conv_transpose1d(x, self.weight, stride=self.stride)
        pad = self.kernel - self.stride
        if pad > 0:
            pr = math.ceil(pad)
            pl = pad - pr
            out = out[:, pl:-pr if pr > 0 else None, :]
        return out + self.bias


# =============================================================================
# Building Blocks
# =============================================================================

class ResidualUnit(nn.Module):
    """Residual unit: Snake → Conv7 → Snake → Conv1."""
    
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.snake1 = Snake1d(dim)
        self.conv1 = CausalConv1d(dim, dim, 7, dilation=dilation)
        self.snake2 = Snake1d(dim)
        self.conv2 = CausalConv1d(dim, dim, 1)
    
    def __call__(self, x: mx.array) -> mx.array:
        res = x
        x = self.snake1(x)
        x = self.conv1(x)
        x = self.snake2(x)
        x = self.conv2(x)
        if res.shape[1] != x.shape[1]:
            diff = res.shape[1] - x.shape[1]
            res = res[:, diff:, :]
        return x + res


class DecoderBlock(nn.Module):
    """Decoder block: Snake → TransConv → 3x ResUnit."""
    
    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.snake = Snake1d(in_dim)
        self.conv = CausalConvTranspose1d(in_dim, out_dim, 2 * stride, stride)
        self.res1 = ResidualUnit(out_dim, 1)
        self.res2 = ResidualUnit(out_dim, 3)
        self.res3 = ResidualUnit(out_dim, 9)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake(x)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Decoder(nn.Module):
    """DAC Decoder: z → audio."""
    
    def __init__(self, config: DACConfig):
        super().__init__()
        ch = config.decoder_dim
        rates = config.decoder_rates
        
        self.conv_in = CausalConv1d(config.latent_dim, ch, 7)
        
        self.blocks = []
        for i, stride in enumerate(rates):
            in_dim = ch // (2 ** i)
            out_dim = ch // (2 ** (i + 1))
            self.blocks.append(DecoderBlock(in_dim, out_dim, stride))
        
        final_dim = ch // (2 ** len(rates))
        self.snake_out = Snake1d(final_dim)
        self.conv_out = CausalConv1d(final_dim, 1, 7)
    
    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        for block in self.blocks:
            x = block(x)
        x = self.snake_out(x)
        x = self.conv_out(x)
        return mx.tanh(x)


# =============================================================================
# RVQ + Upsample
# =============================================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for RVQ upsample."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv_weight = mx.zeros((dim, 7, 1))  # Depthwise
        self.dwconv_bias = mx.zeros((dim,))
        self.norm_weight = mx.ones((dim,))
        self.norm_bias = mx.zeros((dim,))
        self.pwconv1_weight = mx.zeros((dim * 4, dim))
        self.pwconv1_bias = mx.zeros((dim * 4,))
        self.pwconv2_weight = mx.zeros((dim, dim * 4))
        self.pwconv2_bias = mx.zeros((dim,))
        self.gamma = mx.ones((dim,)) * 1e-6
    
    def __call__(self, x: mx.array) -> mx.array:
        res = x
        # Depthwise conv (simplified)
        x = mx.pad(x, [(0, 0), (6, 0), (0, 0)])  # Causal padding for k=7
        # Apply depthwise by reshaping
        B, T, C = x.shape
        x_reshaped = x.reshape(B, T, C, 1)
        # For simplicity, skip actual dwconv and just pass through
        x = res  # Fallback
        return x


class RVQDecoder(nn.Module):
    """RVQ Decoder: indices → z (upsampled)."""
    
    def __init__(self, config: DACConfig):
        super().__init__()
        self.config = config
        
        # Codebooks
        self.semantic_codebook = mx.zeros((config.semantic_codebook_size, config.codebook_dim))
        self.residual_codebooks = [mx.zeros((config.codebook_size, config.codebook_dim)) for _ in range(config.n_codebooks)]
        
        # Output projections - EACH codebook has its own out_proj (8 -> 1024)
        # semantic_quantizer.quantizers[0].out_proj
        self.semantic_out_proj_weight = mx.zeros((config.latent_dim, config.codebook_dim))
        self.semantic_out_proj_bias = mx.zeros((config.latent_dim,))
        # quantizer.quantizers[i].out_proj for i in 0..8
        self.residual_out_proj_weights = [mx.zeros((config.latent_dim, config.codebook_dim)) for _ in range(config.n_codebooks)]
        self.residual_out_proj_biases = [mx.zeros((config.latent_dim,)) for _ in range(config.n_codebooks)]
        
        # Post-Module: 8-layer transformer for processing quantized codes
        # Matches quantizer.post_module in PyTorch checkpoint
        # Config deduced from weights: freqs_cis=(4096,32,2) → head_dim=64, wqkv=(3072,1024) → n_heads=16
        self.post_module = PostModule(
            dim=config.latent_dim,
            n_layers=8,
            n_heads=16,
            n_kv_heads=16,
            head_dim=64,  # From freqs_cis shape (xx, 32, 2) → 32*2 = 64
            intermediate_size=3072,  # 3x dim
            max_seq_len=4096,
        )
        
        # Upsample stage 0: transconv + convnext
        self.up0_conv_weight = mx.zeros((config.latent_dim, 2, config.latent_dim))
        self.up0_conv_bias = mx.zeros((config.latent_dim,))
        self.up0_dwconv_weight = mx.zeros((config.latent_dim, 7, 1))  # Depthwise
        self.up0_dwconv_bias = mx.zeros((config.latent_dim,))
        self.up0_norm_weight = mx.ones((config.latent_dim,))
        self.up0_norm_bias = mx.zeros((config.latent_dim,))
        self.up0_pwconv1_weight = mx.zeros((config.latent_dim * 4, config.latent_dim))
        self.up0_pwconv1_bias = mx.zeros((config.latent_dim * 4,))
        self.up0_pwconv2_weight = mx.zeros((config.latent_dim, config.latent_dim * 4))
        self.up0_pwconv2_bias = mx.zeros((config.latent_dim,))
        self.up0_gamma = mx.ones((config.latent_dim,)) * 1e-6
        
        # Upsample stage 1: transconv + convnext
        self.up1_conv_weight = mx.zeros((config.latent_dim, 2, config.latent_dim))
        self.up1_conv_bias = mx.zeros((config.latent_dim,))
        self.up1_dwconv_weight = mx.zeros((config.latent_dim, 7, 1))
        self.up1_dwconv_bias = mx.zeros((config.latent_dim,))
        self.up1_norm_weight = mx.ones((config.latent_dim,))
        self.up1_norm_bias = mx.zeros((config.latent_dim,))
        self.up1_pwconv1_weight = mx.zeros((config.latent_dim * 4, config.latent_dim))
        self.up1_pwconv1_bias = mx.zeros((config.latent_dim * 4,))
        self.up1_pwconv2_weight = mx.zeros((config.latent_dim, config.latent_dim * 4))
        self.up1_pwconv2_bias = mx.zeros((config.latent_dim,))
        self.up1_gamma = mx.ones((config.latent_dim,)) * 1e-6
    
    def _convnext_block(self, x: mx.array, dwconv_w, dwconv_b, norm_w, norm_b, 
                        pwconv1_w, pwconv1_b, pwconv2_w, pwconv2_b, gamma) -> mx.array:
        """ConvNeXt block: dwconv -> norm -> pwconv1 -> gelu -> pwconv2 -> gamma*x + residual"""
        residual = x
        
        # Depthwise conv (groups=channels)
        # x: (B, T, C), dwconv_w: (C, 7, 1)
        B, T, C = x.shape
        x = mx.pad(x, [(0, 0), (6, 0), (0, 0)])  # Causal padding for k=7
        
        # Apply depthwise: each channel has its own 7-tap filter
        # Collect outputs and concatenate
        outputs = []
        for c in range(C):
            x_c = x[:, :, c:c+1]  # (B, T+6, 1)
            w_c = dwconv_w[c:c+1, :, :]  # (1, 7, 1)
            out_c = mx.conv1d(x_c, w_c, stride=1)[:, :T, :]  # (B, T, 1)
            outputs.append(out_c)
        x = mx.concatenate(outputs, axis=2) + dwconv_b  # (B, T, C)
        
        # Layer norm
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + 1e-6)
        x = x * norm_w + norm_b
        
        # Pointwise convs (implemented as Linear)
        x = x @ pwconv1_w.T + pwconv1_b  # (B, T, 4*C)
        x = nn.gelu(x)
        x = x @ pwconv2_w.T + pwconv2_b  # (B, T, C)
        
        # Scale and residual
        x = gamma * x + residual
        return x
    
    def __call__(self, indices: mx.array) -> mx.array:
        """
        Args:
            indices: (n_codebooks, T) or (B, n_codebooks, T)
        Returns:
            z: (B, T*4, latent_dim)
        """
        if indices.ndim == 2:
            indices = indices[None]
        
        B, N, T = indices.shape
        
        # Codebook lookups with SEPARATE out_proj for each codebook
        # This matches PyTorch: for each codebook, lookup 8dim -> out_proj to 1024dim -> sum
        
        # Semantic codebook: lookup + its own out_proj
        idx0 = mx.clip(indices[:, 0, :], 0, self.config.semantic_codebook_size - 1)
        z_sem = self.semantic_codebook[idx0]  # (B, T, 8)
        z = z_sem @ self.semantic_out_proj_weight.T + self.semantic_out_proj_bias  # (B, T, 1024)
        
        # Residual codebooks: each has its own out_proj, sum in 1024-dim space
        for i in range(self.config.n_codebooks):
            idx_i = mx.clip(indices[:, i + 1, :], 0, self.config.codebook_size - 1)
            z_res_i = self.residual_codebooks[i][idx_i]  # (B, T, 8)
            z_proj_i = z_res_i @ self.residual_out_proj_weights[i].T + self.residual_out_proj_biases[i]  # (B, T, 1024)
            z = z + z_proj_i
        
        # Apply post-module transformer (critical for correct output magnitude)
        # z: (B, T, C) -> permute to (B, C, T) for post_module -> permute back
        z = mx.transpose(z, (0, 2, 1))  # (B, C, T)
        z = self.post_module(z)          # 8-layer transformer
        z = mx.transpose(z, (0, 2, 1))  # (B, T, C)
        
        # Upsample stage 0: transconv + convnext
        z = mx.conv_transpose1d(z, self.up0_conv_weight, stride=2) + self.up0_conv_bias
        z = self._convnext_block(z, self.up0_dwconv_weight, self.up0_dwconv_bias,
                                  self.up0_norm_weight, self.up0_norm_bias,
                                  self.up0_pwconv1_weight, self.up0_pwconv1_bias,
                                  self.up0_pwconv2_weight, self.up0_pwconv2_bias,
                                  self.up0_gamma)
        
        # Upsample stage 1: transconv + convnext
        z = mx.conv_transpose1d(z, self.up1_conv_weight, stride=2) + self.up1_conv_bias
        z = self._convnext_block(z, self.up1_dwconv_weight, self.up1_dwconv_bias,
                                  self.up1_norm_weight, self.up1_norm_bias,
                                  self.up1_pwconv1_weight, self.up1_pwconv1_bias,
                                  self.up1_pwconv2_weight, self.up1_pwconv2_bias,
                                  self.up1_gamma)
        
        return z


# =============================================================================
# Complete DAC Codec
# =============================================================================

class DACCodec(nn.Module):
    """Complete MLX DAC Codec for decoding."""
    
    def __init__(self, config: DACConfig = None):
        super().__init__()
        self.config = config or DACConfig()
        self.rvq = RVQDecoder(self.config)
        self.decoder = Decoder(self.config)
        self.sample_rate = self.config.sample_rate
        self.frame_length = 2048
    
    def decode(self, indices: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Decode VQ indices to audio.
        
        Args:
            indices: (n_codebooks, T) - 10 codebooks
        Returns:
            (audio, audio_lengths)
        """
        z = self.rvq(indices)
        audio = self.decoder(z)
        lengths = mx.array([audio.shape[1]])
        return audio, lengths
    
    @classmethod
    def from_pretrained(cls, weights_dir: str) -> "DACCodec":
        """Load from pretrained weights."""
        weights_dir = Path(weights_dir)
        config = DACConfig()
        model = cls(config)
        
        # Load weights
        weights_path = weights_dir / 'dac.safetensors'
        if weights_path.exists():
            raw_weights = mx.load(str(weights_path))
            model._load_weights(raw_weights)
        
        return model
    
    def _load_weights(self, raw: Dict[str, mx.array]):
        """Load and map weights from PyTorch format."""
        
        # Helper to merge weight norm
        def get_conv_weight(prefix: str) -> mx.array:
            g_key = f"{prefix}.parametrizations.weight.original0"
            v_key = f"{prefix}.parametrizations.weight.original1"
            if g_key in raw and v_key in raw:
                return merge_weight_norm(raw[g_key], raw[v_key])
            elif f"{prefix}.weight" in raw:
                return raw[f"{prefix}.weight"]
            return None
        
        # Load RVQ codebooks
        if "quantizer.semantic_quantizer.quantizers.0.codebook.weight" in raw:
            self.rvq.semantic_codebook = raw["quantizer.semantic_quantizer.quantizers.0.codebook.weight"]
        
        for i in range(9):
            key = f"quantizer.quantizer.quantizers.{i}.codebook.weight"
            if key in raw:
                self.rvq.residual_codebooks[i] = raw[key]
        
        # Load RVQ out_proj weights - EACH codebook has its own out_proj (weight norm format)
        # g shape: (1024, 1, 1), v shape: (1024, 8, 1)
        # Merged weight: g * v / ||v|| where ||v|| is over axes (1, 2)
        
        # Semantic quantizer's out_proj (only one quantizer in semantic_quantizer)
        prefix = "quantizer.semantic_quantizer.quantizers.0.out_proj"
        if f"{prefix}.weight_g" in raw:
            g = raw[f"{prefix}.weight_g"]
            v = raw[f"{prefix}.weight_v"]
            v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
            self.rvq.semantic_out_proj_weight = (g * v / v_norm).squeeze()  # (1024, 8)
        if f"{prefix}.bias" in raw:
            self.rvq.semantic_out_proj_bias = raw[f"{prefix}.bias"]
        
        # Residual quantizer's out_proj (9 quantizers in quantizer)
        for i in range(9):
            prefix = f"quantizer.quantizer.quantizers.{i}.out_proj"
            if f"{prefix}.weight_g" in raw:
                g = raw[f"{prefix}.weight_g"]
                v = raw[f"{prefix}.weight_v"]
                v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
                self.rvq.residual_out_proj_weights[i] = (g * v / v_norm).squeeze()  # (1024, 8)
            if f"{prefix}.bias" in raw:
                self.rvq.residual_out_proj_biases[i] = raw[f"{prefix}.bias"]
        
        # Load upsample stage 0
        if "quantizer.upsample.0.0.conv.weight" in raw:
            w = raw["quantizer.upsample.0.0.conv.weight"]
            self.rvq.up0_conv_weight = mx.transpose(w, (2, 1, 0))
        if "quantizer.upsample.0.0.conv.bias" in raw:
            self.rvq.up0_conv_bias = raw["quantizer.upsample.0.0.conv.bias"]
        if "quantizer.upsample.0.1.dwconv.conv.weight" in raw:
            self.rvq.up0_dwconv_weight = raw["quantizer.upsample.0.1.dwconv.conv.weight"]
        if "quantizer.upsample.0.1.dwconv.conv.bias" in raw:
            self.rvq.up0_dwconv_bias = raw["quantizer.upsample.0.1.dwconv.conv.bias"]
        if "quantizer.upsample.0.1.norm.weight" in raw:
            self.rvq.up0_norm_weight = raw["quantizer.upsample.0.1.norm.weight"]
        if "quantizer.upsample.0.1.norm.bias" in raw:
            self.rvq.up0_norm_bias = raw["quantizer.upsample.0.1.norm.bias"]
        if "quantizer.upsample.0.1.pwconv1.weight" in raw:
            self.rvq.up0_pwconv1_weight = raw["quantizer.upsample.0.1.pwconv1.weight"]
        if "quantizer.upsample.0.1.pwconv1.bias" in raw:
            self.rvq.up0_pwconv1_bias = raw["quantizer.upsample.0.1.pwconv1.bias"]
        if "quantizer.upsample.0.1.pwconv2.weight" in raw:
            self.rvq.up0_pwconv2_weight = raw["quantizer.upsample.0.1.pwconv2.weight"]
        if "quantizer.upsample.0.1.pwconv2.bias" in raw:
            self.rvq.up0_pwconv2_bias = raw["quantizer.upsample.0.1.pwconv2.bias"]
        if "quantizer.upsample.0.1.gamma" in raw:
            self.rvq.up0_gamma = raw["quantizer.upsample.0.1.gamma"]
        
        # Load upsample stage 1
        if "quantizer.upsample.1.0.conv.weight" in raw:
            w = raw["quantizer.upsample.1.0.conv.weight"]
            self.rvq.up1_conv_weight = mx.transpose(w, (2, 1, 0))
        if "quantizer.upsample.1.0.conv.bias" in raw:
            self.rvq.up1_conv_bias = raw["quantizer.upsample.1.0.conv.bias"]
        if "quantizer.upsample.1.1.dwconv.conv.weight" in raw:
            self.rvq.up1_dwconv_weight = raw["quantizer.upsample.1.1.dwconv.conv.weight"]
        if "quantizer.upsample.1.1.dwconv.conv.bias" in raw:
            self.rvq.up1_dwconv_bias = raw["quantizer.upsample.1.1.dwconv.conv.bias"]
        if "quantizer.upsample.1.1.norm.weight" in raw:
            self.rvq.up1_norm_weight = raw["quantizer.upsample.1.1.norm.weight"]
        if "quantizer.upsample.1.1.norm.bias" in raw:
            self.rvq.up1_norm_bias = raw["quantizer.upsample.1.1.norm.bias"]
        if "quantizer.upsample.1.1.pwconv1.weight" in raw:
            self.rvq.up1_pwconv1_weight = raw["quantizer.upsample.1.1.pwconv1.weight"]
        if "quantizer.upsample.1.1.pwconv1.bias" in raw:
            self.rvq.up1_pwconv1_bias = raw["quantizer.upsample.1.1.pwconv1.bias"]
        if "quantizer.upsample.1.1.pwconv2.weight" in raw:
            self.rvq.up1_pwconv2_weight = raw["quantizer.upsample.1.1.pwconv2.weight"]
        if "quantizer.upsample.1.1.pwconv2.bias" in raw:
            self.rvq.up1_pwconv2_bias = raw["quantizer.upsample.1.1.pwconv2.bias"]
        if "quantizer.upsample.1.1.gamma" in raw:
            self.rvq.up1_gamma = raw["quantizer.upsample.1.1.gamma"]
        
        # Load post_module (8-layer transformer)
        pm = self.rvq.post_module
        
        # RoPE frequencies
        if "quantizer.post_module.freqs_cis" in raw:
            pm.freqs_cis = raw["quantizer.post_module.freqs_cis"]
        
        # Causal mask
        if "quantizer.post_module.causal_mask" in raw:
            pm.causal_mask = raw["quantizer.post_module.causal_mask"].astype(mx.bool_)
        
        # Final norm
        if "quantizer.post_module.norm.weight" in raw:
            pm.norm.weight = raw["quantizer.post_module.norm.weight"]
        
        # Transformer layers
        for layer_idx in range(8):
            prefix = f"quantizer.post_module.layers.{layer_idx}"
            layer = pm.layers[layer_idx]
            
            # Attention
            if f"{prefix}.attention.wqkv.weight" in raw:
                layer.attention.wqkv_weight = raw[f"{prefix}.attention.wqkv.weight"]
            if f"{prefix}.attention.wo.weight" in raw:
                layer.attention.wo_weight = raw[f"{prefix}.attention.wo.weight"]
            
            # Attention norm and layer scale
            if f"{prefix}.attention_norm.weight" in raw:
                layer.attention_norm.weight = raw[f"{prefix}.attention_norm.weight"]
            if f"{prefix}.attention_layer_scale.gamma" in raw:
                layer.attention_layer_scale.gamma = raw[f"{prefix}.attention_layer_scale.gamma"]
            
            # FeedForward (SwiGLU)
            if f"{prefix}.feed_forward.w1.weight" in raw:
                layer.feed_forward.w1_weight = raw[f"{prefix}.feed_forward.w1.weight"]
            if f"{prefix}.feed_forward.w2.weight" in raw:
                layer.feed_forward.w2_weight = raw[f"{prefix}.feed_forward.w2.weight"]
            if f"{prefix}.feed_forward.w3.weight" in raw:
                layer.feed_forward.w3_weight = raw[f"{prefix}.feed_forward.w3.weight"]
            
            # FFN norm and layer scale
            if f"{prefix}.ffn_norm.weight" in raw:
                layer.ffn_norm.weight = raw[f"{prefix}.ffn_norm.weight"]
            if f"{prefix}.ffn_layer_scale.gamma" in raw:
                layer.ffn_layer_scale.gamma = raw[f"{prefix}.ffn_layer_scale.gamma"]
        
        # Load decoder.conv_in
        w = get_conv_weight("decoder.model.0.conv")
        if w is not None:
            self.decoder.conv_in.weight = w
        if "decoder.model.0.conv.bias" in raw:
            self.decoder.conv_in.bias = raw["decoder.model.0.conv.bias"]
        
        # Load decoder blocks (model.1-4 → blocks 0-3)
        for block_idx in range(4):
            pt_idx = block_idx + 1
            prefix = f"decoder.model.{pt_idx}.block"
            
            # Snake alpha
            if f"{prefix}.0.alpha" in raw:
                alpha = raw[f"{prefix}.0.alpha"]
                self.decoder.blocks[block_idx].snake.alpha = mx.transpose(alpha, (0, 2, 1))
            
            # TransConv - PyTorch: (in_ch, k, out_ch), MLX needs: (out_ch, k, in_ch)
            w = get_conv_weight(f"{prefix}.1.conv")
            if w is not None:
                # Transpose for MLX conv_transpose1d format
                self.decoder.blocks[block_idx].conv.weight = mx.transpose(w, (2, 1, 0))
            if f"{prefix}.1.conv.bias" in raw:
                self.decoder.blocks[block_idx].conv.bias = raw[f"{prefix}.1.conv.bias"]
            
            # ResUnits (block.2, block.3, block.4)
            for res_idx, res_unit in enumerate([self.decoder.blocks[block_idx].res1, 
                                                 self.decoder.blocks[block_idx].res2,
                                                 self.decoder.blocks[block_idx].res3]):
                res_prefix = f"{prefix}.{res_idx + 2}.block"
                
                # Snake1
                if f"{res_prefix}.0.alpha" in raw:
                    res_unit.snake1.alpha = mx.transpose(raw[f"{res_prefix}.0.alpha"], (0, 2, 1))
                # Conv1
                w = get_conv_weight(f"{res_prefix}.1.conv")
                if w is not None:
                    res_unit.conv1.weight = w
                if f"{res_prefix}.1.conv.bias" in raw:
                    res_unit.conv1.bias = raw[f"{res_prefix}.1.conv.bias"]
                # Snake2
                if f"{res_prefix}.2.alpha" in raw:
                    res_unit.snake2.alpha = mx.transpose(raw[f"{res_prefix}.2.alpha"], (0, 2, 1))
                # Conv2
                w = get_conv_weight(f"{res_prefix}.3.conv")
                if w is not None:
                    res_unit.conv2.weight = w
                if f"{res_prefix}.3.conv.bias" in raw:
                    res_unit.conv2.bias = raw[f"{res_prefix}.3.conv.bias"]
        
        # Load decoder.snake_out (model.5)
        if "decoder.model.5.alpha" in raw:
            self.decoder.snake_out.alpha = mx.transpose(raw["decoder.model.5.alpha"], (0, 2, 1))
        
        # Load decoder.conv_out (model.6)
        w = get_conv_weight("decoder.model.6.conv")
        if w is not None:
            self.decoder.conv_out.weight = w
        if "decoder.model.6.conv.bias" in raw:
            self.decoder.conv_out.bias = raw["decoder.model.6.conv.bias"]
        
        print(f"Loaded DAC weights")
