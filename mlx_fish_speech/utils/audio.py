"""
Audio I/O utilities for Fish-Speech.
"""

from pathlib import Path
from typing import Union, Tuple

import numpy as np


def load_audio(
    path: Union[str, Path],
    sample_rate: int = 44100,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate (resamples if different)
        mono: Convert to mono if stereo
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import soundfile as sf
    
    audio, sr = sf.read(str(path), dtype='float32')
    
    # Convert to mono if needed
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int = 44100,
) -> None:
    """
    Save audio to file.
    
    Args:
        audio: Audio array (float32, normalized to [-1, 1])
        path: Output path
        sample_rate: Sample rate
    """
    import soundfile as sf
    
    # Ensure audio is in valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    # Convert to float32 if not already
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    sf.write(str(path), audio, sample_rate)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Input audio array
        target_db: Target RMS level in dB
        
    Returns:
        Normalized audio array
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)


def pad_or_trim(audio: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or trim audio to exact length.
    
    Args:
        audio: Input audio array
        length: Target length in samples
        pad_value: Value to use for padding
        
    Returns:
        Audio array of exact length
    """
    if len(audio) > length:
        return audio[:length]
    elif len(audio) < length:
        pad_len = length - len(audio)
        return np.pad(audio, (0, pad_len), constant_values=pad_value)
    return audio
