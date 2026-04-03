"""
PyTorch to MLX weight conversion for Fish-Speech.

Usage:
    python -m mlx_fish_speech.convert \
        --input /path/to/fish-speech/checkpoints/openaudio-s1-mini \
        --output /path/to/mlx-fish-speech/weights
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import mlx.core as mx
import numpy as np


def convert_key(key: str) -> str:
    """
    Convert PyTorch weight key to MLX format.
    
    Fish-Speech uses mostly compatible naming, but some adjustments needed.
    """
    # Map PyTorch naming to MLX naming
    replacements = [
        # Attention projections
        ("wqkv.", "wqkv."),
        ("wo.", "wo."),
        # Fast decoder naming
        ("fast_layers.", "fast_layers."),
        # Embeddings
        ("embeddings.word_embeddings.", "token_embedding."),
        ("semantic_token_embedding.", "codebook_embedding."),
    ]
    
    for old, new in replacements:
        key = key.replace(old, new)
    
    return key


def convert_tensor(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    # Move to CPU and convert to numpy
    np_array = tensor.detach().cpu().float().numpy()
    return mx.array(np_array)


def convert_text2semantic_weights(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, mx.array]:
    """
    Convert Text2Semantic (DualARTransformer) weights.
    
    Maps from Fish-Speech PyTorch format to MLX format.
    """
    mlx_weights = {}
    
    for key, tensor in state_dict.items():
        # Skip non-weight keys
        if "num_batches_tracked" in key:
            continue
        
        new_key = convert_key(key)
        mlx_weights[new_key] = convert_tensor(tensor)
        
    return mlx_weights


def convert_dac_weights(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, mx.array]:
    """
    Convert DAC Codec weights.
    
    Handles Conv1d weight transposition for MLX format.
    """
    mlx_weights = {}
    
    for key, tensor in state_dict.items():
        # Skip non-weight keys
        if "num_batches_tracked" in key:
            continue
        
        # Conv1d weights need transposition
        # PyTorch: [out_channels, in_channels, kernel_size]
        # MLX: [out_channels, kernel_size, in_channels]
        if "conv" in key and "weight" in key and tensor.dim() == 3:
            tensor = tensor.transpose(1, 2)
        
        new_key = convert_key(key)
        mlx_weights[new_key] = convert_tensor(tensor)
    
    return mlx_weights


def convert_model(input_path: str, output_path: str) -> None:
    """
    Convert Fish-Speech model from PyTorch to MLX format.
    
    Args:
        input_path: Path to Fish-Speech checkpoint directory
        output_path: Path to save MLX weights
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting from: {input_path}")
    print(f"Converting to: {output_path}")
    
    # Load config
    config_path = input_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Model config: {config.get('model_type', 'unknown')}")
    print(f"  - dim: {config.get('dim')}")
    print(f"  - n_layer: {config.get('n_layer')}")
    print(f"  - vocab_size: {config.get('vocab_size')}")
    
    # Load PyTorch weights
    model_path = input_path / "model.pth"
    print(f"\nLoading PyTorch weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    
    print(f"Found {len(state_dict)} weight tensors")
    
    # Convert Text2Semantic weights
    print("\nConverting Text2Semantic weights...")
    text2semantic_weights = convert_text2semantic_weights(state_dict)
    
    # Save as safetensors
    text2semantic_path = output_path / "text2semantic.safetensors"
    mx.save_safetensors(str(text2semantic_path), text2semantic_weights)
    print(f"Saved {len(text2semantic_weights)} tensors to {text2semantic_path}")
    
    # Load and convert DAC codec weights
    codec_path = input_path / "codec.pth"
    if codec_path.exists():
        print(f"\nLoading DAC codec weights from {codec_path}...")
        codec_state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        
        # Handle nested state_dict
        if "state_dict" in codec_state_dict:
            codec_state_dict = codec_state_dict["state_dict"]
        
        print(f"Found {len(codec_state_dict)} codec weight tensors")
        
        # Convert DAC weights
        print("Converting DAC codec weights...")
        dac_weights = convert_dac_weights(codec_state_dict)
        
        # Save as safetensors
        dac_path = output_path / "dac.safetensors"
        mx.save_safetensors(str(dac_path), dac_weights)
        print(f"Saved {len(dac_weights)} tensors to {dac_path}")
    
    # Copy config
    output_config = output_path / "config.json"
    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nCopied config to {output_config}")
    
    # Copy tokenizer if exists
    tokenizer_path = input_path / "tokenizer.tiktoken"
    if tokenizer_path.exists():
        import shutil
        shutil.copy(tokenizer_path, output_path / "tokenizer.tiktoken")
        print(f"Copied tokenizer to {output_path / 'tokenizer.tiktoken'}")
    
    print("\n✅ Conversion complete!")
    print(f"MLX weights saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Fish-Speech PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to Fish-Speech checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save MLX weights"
    )
    
    args = parser.parse_args()
    convert_model(args.input, args.output)


if __name__ == "__main__":
    main()
