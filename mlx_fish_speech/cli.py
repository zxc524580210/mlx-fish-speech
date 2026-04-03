"""
Command-line interface for MLX Fish-Speech.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="MLX Fish-Speech: Text-to-Speech using MLX for Apple Silicon"
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="fishaudio/openaudio-s1-mini",
        help="Model path or HuggingFace model ID"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading model: {args.model}")
    
    try:
        from mlx_fish_speech import FishSpeech
        from mlx_fish_speech.generate import GenerationConfig
        
        # Load model
        tts = FishSpeech.from_pretrained(args.model)
        
        if args.verbose:
            print(f"Generating speech for: {args.text[:50]}...")
        
        # Generate audio
        config = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        
        audio = tts.generate(args.text, config)
        
        # Save output
        audio.save(args.output)
        
        if args.verbose:
            print(f"Saved audio to: {args.output}")
            print(f"Duration: {audio.duration:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
