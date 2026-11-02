"""Shared pytest fixtures for the AI voice vishing detection test suite.

Since the raw ASVspoof audio data is large and not always available,
most tests use synthetic audio generated on the fly.  An integration
fixture (``synthetic_audio_dir``) can create a fake directory tree
with FLAC files for end-to-end pipeline tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SYNTHETIC AUDIO GENERATORS
# ============================================================

def generate_bonafide_audio(
    duration: float = 3.0,
    sr: int = 16000,
    seed: int = 0,
) -> np.ndarray:
    """Generate audio that resembles natural speech (harmonic + noise).

    Not a real speech signal, but has the spectral richness needed for
    meaningful feature extraction.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    # Fundamental frequency with slight modulation
    f0 = 120 + 10 * np.sin(2 * np.pi * 3 * t)
    # Harmonics
    audio = np.zeros_like(t)
    for h in range(1, 6):
        audio += (1.0 / h) * np.sin(2 * np.pi * h * f0 * t)

    # Add breath noise
    audio += 0.05 * rng.standard_normal(len(t))

    # Amplitude envelope (simulate syllables)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    audio *= envelope

    # Normalise
    audio = audio / np.max(np.abs(audio)) * 0.9
    return audio.astype(np.float32)


def generate_spoof_audio(
    duration: float = 3.0,
    sr: int = 16000,
    seed: int = 100,
) -> np.ndarray:
    """Generate audio that resembles synthetic / spoofed speech.

    Uses a more regular spectral structure and subtle artifacts
    that differ from the bonafide generator.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    # More regular F0 with digital quantization artifacts
    f0 = 120 + 5 * np.sin(2 * np.pi * 3 * t)
    f0 = np.round(f0 / 5) * 5  # quantize — creates artifacts

    audio = np.zeros_like(t)
    for h in range(1, 8):
        audio += (0.8 / h) * np.sin(2 * np.pi * h * f0 * t)

    # Less natural noise
    audio += 0.02 * rng.standard_normal(len(t))

    # Flat-ish envelope
    envelope = 0.7 * np.ones_like(t)
    audio *= envelope

    audio = audio / np.max(np.abs(audio)) * 0.9
    return audio.astype(np.float32)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def bonafide_audio():
    """A 3-second bonafide-like audio at 16 kHz."""
    return generate_bonafide_audio(duration=3.0, sr=16000, seed=42)


@pytest.fixture
def spoof_audio():
    """A 3-second spoof-like audio at 16 kHz."""
    return generate_spoof_audio(duration=3.0, sr=16000, seed=142)


@pytest.fixture
def sample_rate():
    return 16000


@pytest.fixture
def audio_pair(bonafide_audio, spoof_audio):
    """Tuple of (bonafide, spoof) audio arrays."""
    return bonafide_audio, spoof_audio


@pytest.fixture
def known_labels_scores():
    """Known labels and scores for metrics testing.

    Labels: 0 = bonafide, 1 = spoof
    Scores: higher = more likely spoof
    """
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Spoofs have higher scores (correctly classified)
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.7, 0.8, 0.75, 0.85, 0.65])
    return labels, scores


@pytest.fixture
def mixed_labels_scores():
    """Labels and scores with some misclassifications."""
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Two bonafides scored high (FP), one spoof scored low (FN)
    scores = np.array([0.1, 0.2, 0.8, 0.15, 0.3, 0.7, 0.25, 0.8, 0.85, 0.9])
    return labels, scores


@pytest.fixture
def synthetic_train_data():
    """Flat feature matrix and labels for model training tests.

    Creates 60 bonafide + 60 spoof samples with distinct feature
    patterns so models can learn the separation.
    """
    rng = np.random.default_rng(42)

    n_bona = 60
    n_spoof = 60
    n_features = 20

    # Bonafide: mean ~ 0, std ~ 1
    X_bona = rng.normal(0, 1, (n_bona, n_features))
    # Spoof: mean ~ 1, std ~ 1
    X_spoof = rng.normal(1, 1, (n_spoof, n_features))

    X = np.vstack([X_bona, X_spoof]).astype(np.float32)
    y = np.concatenate([np.zeros(n_bona), np.ones(n_spoof)]).astype(int)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def synthetic_eval_data():
    """Flat feature matrix and labels for model evaluation tests."""
    rng = np.random.default_rng(99)

    n_bona = 30
    n_spoof = 30
    n_features = 20

    X_bona = rng.normal(0, 1, (n_bona, n_features))
    X_spoof = rng.normal(1, 1, (n_spoof, n_features))

    X = np.vstack([X_bona, X_spoof]).astype(np.float32)
    y = np.concatenate([np.zeros(n_bona), np.ones(n_spoof)]).astype(int)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def synthetic_audio_dir(tmp_path):
    """Create a fake ASVspoof directory tree with synthetic FLAC files.

    Returns
    -------
    Path  base directory (e.g. tmp_path / "data" / "raw" / "LA" / "LA")
    """
    base = tmp_path / "data" / "raw" / "LA" / "LA"

    splits = {
        "train": "ASVspoof2019_LA_train",
        "dev": "ASVspoof2019_LA_dev",
        "eval": "ASVspoof2019_LA_eval",
    }

    for split_name, folder in splits.items():
        flac_dir = base / folder / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)

        # 5 bonafide + 5 spoof per split
        for i in range(5):
            fname = f"LA_{split_name[0].upper()}_BONAFIDE_{i:04d}.flac"
            audio = generate_bonafide_audio(duration=3.0, sr=16000, seed=i)
            sf.write(flac_dir / fname, audio, 16000)

        for i in range(5):
            fname = f"LA_{split_name[0].upper()}_SPOOF_{i:04d}.flac"
            audio = generate_spoof_audio(duration=3.0, sr=16000, seed=100 + i)
            sf.write(flac_dir / fname, audio, 16000)

    return base


@pytest.fixture
def synthetic_metadata(synthetic_audio_dir):
    """Create a metadata DataFrame matching the synthetic audio dir."""
    rows = []

    splits = {
        "train": "ASVspoof2019_LA_train",
        "dev": "ASVspoof2019_LA_dev",
        "eval": "ASVspoof2019_LA_eval",
    }

    for split_name, folder in splits.items():
        flac_dir = synthetic_audio_dir / folder / "flac"
        for f in sorted(flac_dir.glob("*.flac")):
            is_spoof = "SPOOF" in f.name
            rows.append({
                "file": f.stem,
                "file_path": str(f),
                "split": split_name,
                "speaker_id": f"LA_{split_name[0].upper()}",
                "system_id": "-" if not is_spoof else "A01",
                "label": "spoof" if is_spoof else "bonafide",
                "duration_sec": 3.0,
                "sample_rate": 16000,
                "channels": 1,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def config_with_synthetic_audio(synthetic_audio_dir, tmp_path):
    """Config pointing to synthetic audio directory."""
    from src.utils.config import Config

    config = Config()
    config.raw_audio_base = synthetic_audio_dir
    config.metadata_dir = tmp_path / "metadata"
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = tmp_path / "outputs"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config
