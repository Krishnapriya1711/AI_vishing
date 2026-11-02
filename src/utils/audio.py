"""Audio loading and telephone-channel simulation utilities.

This module provides a clean, importable interface for loading ASVspoof
audio files with optional telephone band-limiting and G.711 A-law codec
degradation, mirroring the logic in ``preprocessing/audio_loader.py``
but with configurable paths and proper Linux path handling.
"""

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt


# ============================================================
# FILE LOOKUP
# ============================================================

_SPLIT_FOLDERS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}


def find_audio_file(
    file_name: str,
    split: str,
    base_dir: Path,
) -> Path:
    """Locate a FLAC audio file in the ASVspoof directory tree.

    Parameters
    ----------
    file_name : str
        Utterance ID, with or without ``.flac`` extension.
    split : str
        One of ``train``, ``dev``, ``eval``.
    base_dir : Path
        Root of the ASVspoof data tree (e.g. ``data/raw/LA/LA``).

    Returns
    -------
    Path
        Full path to the audio file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if split not in _SPLIT_FOLDERS:
        raise ValueError(f"Unknown split: {split!r}")

    audio_dir = Path(base_dir) / _SPLIT_FOLDERS[split] / "flac"

    name = str(file_name)
    if not name.lower().endswith(".flac"):
        name += ".flac"

    path = audio_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    return path


# ============================================================
# TELEPHONE CHANNEL SIMULATION
# ============================================================

def telephone_filter(
    audio: np.ndarray,
    sr: int,
    low: float = 300.0,
    high: float = 3400.0,
) -> np.ndarray:
    """Apply a 6th-order Butterworth band-pass filter mimicking
    the telephone speech band (300–3400 Hz)."""
    nyq = sr / 2.0
    if high >= nyq:
        high = nyq - 1.0
    if low <= 0:
        low = 1.0

    sos = butter(
        6,
        [low, high],
        btype="bandpass",
        fs=sr,
        output="sos",
    )
    return sosfilt(sos, audio)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise audio to 0.95 to prevent clipping."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


# ============================================================
# G.711 A-LAW CODEC
# ============================================================

def alaw_encode(samples: np.ndarray) -> np.ndarray:
    """Encode int16 PCM samples to G.711 A-law bytes."""
    samples = np.asarray(samples, dtype=np.int16)
    encoded = np.zeros(len(samples), dtype=np.uint8)

    for i in range(len(samples)):
        sample = int(samples[i])

        sign = 0x80 if sample < 0 else 0x00
        value = abs(sample)
        if value > 32635:
            value = 32635

        if value >= 256:
            exponent = 7
            mask = 0x4000
            while exponent > 0 and (value & mask) == 0:
                exponent -= 1
                mask >>= 1
            mantissa = (value >> (exponent + 3)) & 0x0F
            alaw = sign | (exponent << 4) | mantissa
        else:
            alaw = sign | (value >> 4)

        encoded[i] = alaw ^ 0x55

    return encoded


def alaw_decode(encoded: np.ndarray) -> np.ndarray:
    """Decode G.711 A-law bytes back to int16 PCM samples."""
    encoded = np.asarray(encoded, dtype=np.uint8)
    decoded = np.zeros(len(encoded), dtype=np.int16)

    for i in range(len(encoded)):
        byte = int(encoded[i])
        byte ^= 0x55

        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F

        if exponent == 0:
            value = (mantissa << 4) + 8
        else:
            value = ((mantissa << 4) + 0x108) << (exponent - 1)

        if sign:
            value = -value

        decoded[i] = np.clip(value, -32768, 32767)

    return decoded


def apply_alaw(audio: np.ndarray) -> np.ndarray:
    """Round-trip audio through G.711 A-law encode → decode.

    Parameters
    ----------
    audio : np.ndarray
        Float audio in roughly [-1, 1].

    Returns
    -------
    np.ndarray
        Codec-degraded float audio.
    """
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    encoded = alaw_encode(audio_int16)
    decoded = alaw_decode(encoded)
    return decoded.astype(np.float32) / 32767.0


# ============================================================
# UNIFIED AUDIO LOADER
# ============================================================

def load_audio(
    file_name: str,
    split: str,
    condition: str = "clean",
    duration: float | None = None,
    start_sec: float = 0.0,
    base_dir: Path | None = None,
    target_sr_clean: int = 16000,
    target_sr_phone: int = 8000,
) -> tuple[np.ndarray, int]:
    """Load an ASVspoof audio file with optional telephone / codec
    degradation and duration cropping.

    Parameters
    ----------
    file_name : str
        Utterance ID (with or without ``.flac``).
    split : str
        ``train``, ``dev``, or ``eval``.
    condition : str
        ``clean``, ``telephone``, or ``telephone_codec``.
    duration : float or None
        Desired duration in seconds (cropped from ``start_sec``).
    start_sec : float
        Start offset in seconds.
    base_dir : Path or None
        ASVspoof data root.  When ``None``, the Config default is used.
    target_sr_clean / target_sr_phone : int
        Sample rates for each condition.

    Returns
    -------
    audio : np.ndarray (float32, mono)
    sr : int
    """
    # Resolve paths lazily to avoid circular import at module load
    if base_dir is None:
        from .config import DEFAULT_CONFIG
        base_dir = DEFAULT_CONFIG.raw_audio_base

    file_path = find_audio_file(file_name, split, base_dir)

    # Load at native sample rate (ASVspoof LA = 16 kHz)
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    # --- Apply condition ---
    if condition == "clean":
        target_sr = target_sr_clean

    elif condition in ("telephone", "telephone_codec"):
        audio = telephone_filter(audio, sr)
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=target_sr_phone,
        )
        sr = target_sr_phone
        audio = normalize(audio)
        target_sr = target_sr_phone

        if condition == "telephone_codec":
            audio = apply_alaw(audio)

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    # --- Crop start ---
    if start_sec > 0:
        start_sample = int(start_sec * sr)
        audio = audio[start_sample:]

    # --- Crop duration ---
    if duration is not None:
        required = int(duration * sr)
        if len(audio) < required:
            raise ValueError(
                f"Audio too short. Requested {duration}s, "
                f"available {len(audio) / sr:.3f}s"
            )
        audio = audio[:required]

    return audio, target_sr
