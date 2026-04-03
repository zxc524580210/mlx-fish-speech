"""
Hybrid E2E TTS with MLX RVQ + PyTorch Decoder.

Uses:
- MLX Text2Semantic (fast generation)
- MLX RVQ codebook lookup (verified exact match with PyTorch)
- PyTorch DAC Decoder (until MLX decoder is fully implemented)

This allows faster overall pipeline by offloading the RVQ lookup to MLX.
"""

import numpy as np
import mlx.core as mx
import torch


class MLXRVQDecoder:
    """MLX-based RVQ codebook lookup."""
    
    def __init__(self, weights_dir: str):
        from mlx_fish_speech.models.dac_weights import load_dac_for_decode
        
        result = load_dac_for_decode(weights_dir)
        self.semantic_codebook = result['semantic_codebook']
        self.residual_codebooks = result['residual_codebooks']
    
    def decode_to_z(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode indices to z (before upsample and decoder).
        
        Args:
            indices: (n_codebooks, T) or (B, n_codebooks, T)
        Returns:
            z: (T, 8) codebook embeddings
        """
        if indices.ndim == 3:
            indices = indices[0]  # Take first batch
        
        indices_mx = mx.array(indices)
        
        # Semantic lookup
        indices_0 = mx.clip(indices_mx[0], 0, 4095)
        z_semantic = self.semantic_codebook[indices_0]
        
        # Residual lookups
        z_residual = mx.zeros_like(z_semantic)
        for i in range(9):
            indices_i = mx.clip(indices_mx[i + 1], 0, 1023)
            z_residual = z_residual + self.residual_codebooks[i][indices_i]
        
        z = z_semantic + z_residual
        return np.array(z)
