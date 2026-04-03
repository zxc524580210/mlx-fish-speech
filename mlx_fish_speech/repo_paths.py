"""Paths anchored to the mlx-fish-speech repository root (parent of `mlx_fish_speech/`)."""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parent


def weights_openaudio_s1_mini() -> Path:
    return REPO_ROOT / "weights" / "openaudio-s1-mini"


def weights_s2_pro() -> Path:
    return REPO_ROOT / "weights" / "s2-pro"
