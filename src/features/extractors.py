"""Feature extraction: MFCC, Mel spectrogram, and pitch / prosody.

Each extractor returns a 2-D array of shape ``(n_frames, n_features)``.
``aggregate_features`` collapses the time axis into a fixed-length vector
(mean + std) for use with non-sequential models (LR, RF, SVM).

For the CNN, the Mel spectrogram can be used directly as a 2-D image.
"""

import numpy as np
import librosa


# ============================================================
# MFCC
# ============================================================

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
    include_deltas: bool = True,
) -> np.ndarray:
    """Extract MFCC features, optionally with delta and delta-delta.

    Returns
    -------
    np.ndarray  shape ``(n_frames, n_mfcc)`` or ``(n_frames, n_mfcc*3)``
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    if include_deltas:
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta1, delta2])
    else:
        features = mfcc

    # (n_frames, n_features)
    return features.T


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 0.0,
    fmax: float | None = None,
    log_scale: bool = True,
) -> np.ndarray:
    """Extract a (log) Mel spectrogram.

    Returns
    -------
    np.ndarray  shape ``(n_mels, n_frames)`` — ready for CNN input.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )

    if log_scale:
        mel = librosa.power_to_db(mel, ref=np.max)

    return mel  # (n_mels, n_frames)


# ============================================================
# PITCH / PROSODY
# ============================================================

def extract_pitch(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 400.0,
) -> np.ndarray:
    """Extract fundamental frequency (F0) contour using pyin.

    Returns
    -------
    np.ndarray  shape ``(n_frames, 1)``
        F0 in Hz.  Unvoiced frames are filled with 0.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    # Replace NaN (unvoiced) with 0
    f0 = np.nan_to_num(f0, nan=0.0)

    return f0.reshape(-1, 1)


# ============================================================
# AGGREGATION (for baseline models)
# ============================================================

def aggregate_features(features: np.ndarray) -> np.ndarray:
    """Collapse a ``(n_frames, n_features)`` array into a fixed-length
    vector using per-feature mean and standard deviation.

    Returns
    -------
    np.ndarray  shape ``(2 * n_features,)``
    """
    if features.shape[0] == 0:
        return np.zeros(2 * features.shape[1], dtype=np.float32)

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


# ============================================================
# UNIFIED EXTRACTION
# ============================================================

def extract_features(
    audio: np.ndarray,
    sr: int,
    feature_type: str = "mfcc",
    aggregate: bool = True,
    **kwargs,
) -> np.ndarray:
    """Extract features by type.

    Parameters
    ----------
    audio : np.ndarray
    sr : int
    feature_type : str
        ``"mfcc"``, ``"mel"``, or ``"pitch"``.
    aggregate : bool
        If True, return a flat vector (mean+std).
        If False, return the raw frame-level matrix (or 2-D spectrogram).

    Returns
    -------
    np.ndarray
        Flat vector if ``aggregate=True``, otherwise 2-D array.
    """
    fmax = kwargs.pop("fmax", None)
    if fmax is None:
        fmax = min(sr / 2, 8000.0)

    if feature_type == "mfcc":
        feats = extract_mfcc(audio, sr, fmax=fmax, **kwargs)

    elif feature_type == "mel":
        mel = extract_mel_spectrogram(audio, sr, fmax=fmax, **kwargs)
        if aggregate:
            # mel is (n_mels, n_frames) → transpose then aggregate
            return aggregate_features(mel.T)
        return mel

    elif feature_type == "pitch":
        feats = extract_pitch(audio, sr, fmax=min(fmax, 400.0), **kwargs)

    else:
        raise ValueError(f"Unknown feature type: {feature_type!r}")

    if aggregate:
        return aggregate_features(feats)
    return feats


# ============================================================
# CNN-READY SPECTROGRAM
# ============================================================

def extract_spectrogram_for_cnn(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 64,
    hop_length: int = 160,
    n_fft: int = 512,
    win_length: int = 400,
    fmin: float = 0.0,
    fmax: float | None = None,
    max_frames: int = 300,
) -> np.ndarray:
    """Extract a fixed-size log-Mel spectrogram for CNN input.

    Pads or truncates the time axis to ``max_frames``.

    Returns
    -------
    np.ndarray  shape ``(n_mels, max_frames)``
    """
    if fmax is None:
        fmax = min(sr / 2, 8000.0)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # Normalise to zero mean, unit variance
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    # Pad or truncate time dimension
    n_frames = mel.shape[1]
    if n_frames < max_frames:
        pad = np.zeros((mel.shape[0], max_frames - n_frames))
        mel = np.concatenate([mel, pad], axis=1)
    else:
        mel = mel[:, :max_frames]

    return mel
