"""
DAC Weight Loading for MLX.

Handles weight norm merging for PyTorch -> MLX conversion.
"""

import math
from pathlib import Path
from typing import Dict

import mlx.core as mx


def merge_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    """
    Merge weight norm parametrization: w = g * v / ||v||
    
    PyTorch weight_norm stores:
      - original0 (g): scale factor, shape (out_ch, 1, 1)
      - original1 (v): normalized weight, shape (out_ch, k, in_ch)
    
    Merged weight = g * v / ||v|| where ||v|| is computed over (1, 2) dims
    """
    # Compute L2 norm of v over kernel and input dims
    v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
    # Merge: w = g * v / ||v||
    return g * v / v_norm


def load_dac_weights(weights_path: str) -> Dict[str, mx.array]:
    """
    Load and convert DAC weights from safetensors.
    
    Handles weight norm merging and key remapping.
    """
    weights = mx.load(str(weights_path))
    converted = {}
    
    # Process weights
    for key, value in weights.items():
        # Skip encoder weights (we only need decode)
        if key.startswith('encoder.'):
            continue
        
        # Handle weight norm parametrizations
        if 'parametrizations.weight.original0' in key:
            # This is 'g' - find corresponding 'v' (original1)
            v_key = key.replace('original0', 'original1')
            if v_key in weights:
                g = value
                v = weights[v_key]
                merged_weight = merge_weight_norm(g, v)
                
                # Create new key without parametrization prefix
                new_key = key.replace('.parametrizations.weight.original0', '.weight')
                converted[new_key] = merged_weight
            continue
        elif 'parametrizations.weight.original1' in key:
            # Skip - handled above with original0
            continue
        
        # Handle weight_g and weight_v (older format)
        if '.weight_g' in key:
            v_key = key.replace('_g', '_v')
            if v_key in weights:
                g = value
                v = weights[v_key]
                # For Linear layers, reshape g,v appropriately
                if g.ndim == 3:  # Conv-style
                    merged = merge_weight_norm(g, v)
                else:  # Linear-style
                    v_norm = mx.sqrt(mx.sum(v * v, axis=-1, keepdims=True) + 1e-12)
                    merged = g * v / v_norm
                new_key = key.replace('.weight_g', '.weight')
                converted[new_key] = merged.squeeze()
            continue
        elif '.weight_v' in key:
            continue
        
        # Regular weights - just copy
        converted[key] = value
    
    return converted


def map_decoder_weights(pt_weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    Map PyTorch decoder weight keys to MLX model structure.
    
    PyTorch structure:
      decoder.model.0 -> conv_in (k=7)
      decoder.model.1 -> block 0 (stride=8)
      decoder.model.2 -> block 1 (stride=8) 
      decoder.model.3 -> block 2 (stride=4)
      decoder.model.4 -> block 3 (stride=2)
      decoder.model.5 -> snake_out
      decoder.model.6 -> conv_out
    
    Block structure:
      block.0 -> snake (alpha)
      block.1 -> conv_trans (transposed conv)
      block.2 -> res1 (residual unit)
      block.3 -> res2
      block.4 -> res3
    
    ResidualUnit structure:
      block.0 -> snake1 (alpha)
      block.1 -> conv1 (k=7, dilated)
      block.2 -> snake2 (alpha)
      block.3 -> conv2 (k=1)
    """
    mlx_weights = {}
    
    for key, value in pt_weights.items():
        if not key.startswith('decoder.'):
            continue
        
        # Transpose conv weights from PyTorch (out, kernel, in) to MLX (in, kernel, out)
        # Actually MLX conv1d expects (out, kernel, in), same as PyTorch for regular conv
        # But for ConvTranspose, PyTorch is (in, kernel, out), MLX is (in, kernel, out) too
        
        # For alpha, squeeze the shape from (1, C, 1) to (1, 1, C)
        if '.alpha' in key:
            value = mx.transpose(value, (0, 2, 1))
        
        # Map to MLX key structure
        new_key = key.replace('decoder.model.', 'decoder.')
        
        # Reindex blocks
        parts = new_key.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx == 0:
                # conv_in
                new_key = 'decoder.conv_in.' + '.'.join(parts[2:])
            elif idx in [1, 2, 3, 4]:
                # decoder blocks
                block_idx = idx - 1
                new_key = f'decoder.blocks.{block_idx}.' + '.'.join(parts[2:])
            elif idx == 5:
                # snake_out
                new_key = 'decoder.snake_out.' + '.'.join(parts[2:])
            elif idx == 6:
                # conv_out  
                new_key = 'decoder.conv_out.' + '.'.join(parts[2:])
        
        mlx_weights[new_key] = value
    
    return mlx_weights


def map_quantizer_weights(pt_weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Map quantizer weights to MLX structure."""
    mlx_weights = {}
    
    for key, value in pt_weights.items():
        if not key.startswith('quantizer.'):
            continue
        
        # For alpha, transpose
        if '.alpha' in key:
            value = mx.transpose(value, (0, 2, 1))
        
        new_key = key
        
        # Map codebooks
        if 'semantic_quantizer.quantizers.0.codebook.weight' in key:
            new_key = 'quantizer.semantic_codebook'
        elif 'quantizer.quantizers.' in key and '.codebook.weight' in key:
            # Extract quantizer index
            idx = int(key.split('quantizers.')[1].split('.')[0])
            new_key = f'quantizer.residual_codebooks.{idx}'
        
        # Map upsample layers
        if 'quantizer.upsample.' in key:
            new_key = key.replace('quantizer.upsample.', 'quantizer.upsample.layers.')
        
        mlx_weights[new_key] = value
    
    return mlx_weights


def load_dac_for_decode(weights_dir: str):
    """
    Load DAC weights for decode-only operation.
    
    Returns tuple of (codebooks, upsample_weights, decoder_weights)
    """
    weights_path = Path(weights_dir) / 'dac.safetensors'
    
    # Load and merge weight norms
    pt_weights = load_dac_weights(str(weights_path))
    
    # Extract relevant parts
    result = {
        'semantic_codebook': None,
        'residual_codebooks': [],
        'upsample': {},
        'decoder': {},
    }
    
    # Semantic codebook
    for k, v in pt_weights.items():
        if 'semantic_quantizer.quantizers.0.codebook' in k:
            result['semantic_codebook'] = v
            break
    
    # Residual codebooks (9 total)
    for i in range(9):
        for k, v in pt_weights.items():
            if f'quantizer.quantizers.{i}.codebook' in k:
                result['residual_codebooks'].append(v)
                break
    
    # Upsample weights
    for k, v in pt_weights.items():
        if 'quantizer.upsample.' in k:
            result['upsample'][k] = v
    
    # Decoder weights
    for k, v in pt_weights.items():
        if k.startswith('decoder.'):
            result['decoder'][k] = v
    
    return result
