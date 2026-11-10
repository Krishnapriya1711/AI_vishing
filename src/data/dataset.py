"""Dataset utilities: metadata loading, feature pre-extraction,
PyTorch ``Dataset`` for the CNN, and flat-array helpers for baselines.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.audio import load_audio
from ..utils.config import DEFAULT_CONFIG
from ..features.extractors import (
    extract_features,
    extract_spectrogram_for_cnn,
    aggregate_features,
    extract_mfcc,
    extract_mel_spectrogram,
    extract_pitch,
)


# ============================================================
# METADATA LOADING
# ============================================================

def load_metadata(
    metadata_name: str = "asvspoof2019_la_audio_stats.csv",
    config=None,
) -> pd.DataFrame:
    """Load a metadata CSV from the metadata directory.

    Handles Windows-style backslash paths by converting to OS-native.
    """
    if config is None:
        config = DEFAULT_CONFIG

    path = Path(config.metadata_dir) / metadata_name
    df = pd.read_csv(path)

    # Normalise file_path separators
    if "file_path" in df.columns:
        df["file_path"] = df["file_path"].str.replace("\\", "/", regex=False)

    return df


def filter_metadata(
    df: pd.DataFrame,
    split: str | None = None,
    condition: str | None = None,
    window_sec: float | None = None,
    max_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Filter metadata by split, condition, window size, and optionally
    subsample.

    Parameters
    ----------
    df : DataFrame
    split : str or None
        ``train``, ``dev``, ``eval``.
    condition : str or None
        ``clean``, ``telephone``, ``telephone_codec``.
    window_sec : float or None
    max_samples : int or None
        Cap total rows (after filtering) for development.
    random_state : int

    Returns
    -------
    DataFrame (filtered copy)
    """
    result = df.copy()

    if split is not None and "split" in result.columns:
        result = result[result["split"] == split]

    if condition is not None and "condition" in result.columns:
        result = result[result["condition"] == condition]

    if window_sec is not None and "window_sec" in result.columns:
        result = result[result["window_sec"] == window_sec]

    if max_samples is not None and len(result) > max_samples:
        # Stratified subsample to keep class balance
        from collections import Counter
        label_counts = Counter(result["label"])
        total = len(result)
        parts = []
        for label, count in label_counts.items():
            n_keep = max(1, int(max_samples * count / total))
            n_keep = min(n_keep, count)
            parts.append(
                result[result["label"] == label].sample(
                    n=n_keep, random_state=random_state,
                )
            )
        result = pd.concat(parts, ignore_index=True)

    return result.reset_index(drop=True)


def get_label(row: pd.Series, label_map: dict | None = None) -> int:
    """Map a textual label to 0 (bonafide) / 1 (spoof)."""
    if label_map is None:
        label_map = DEFAULT_CONFIG.label_map

    text = row.get("label", "spoof")
    return label_map.get(text, 1)


# ============================================================
# FLAT FEATURE EXTRACTION (for baseline models)
# ============================================================

def prepare_flat_features(
    df: pd.DataFrame,
    condition: str = "clean",
    window_sec: float | None = None,
    feature_type: str = "mfcc",
    config=None,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract aggregated features for every row in the metadata.

    Returns
    -------
    X : np.ndarray  (n_samples, n_flat_features)
    y : np.ndarray  (n_samples,)  0 = bonafide, 1 = spoof
    """
    if config is None:
        config = DEFAULT_CONFIG

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        try:
            audio, sr = load_audio(
                file_name=row["file"],
                split=row["split"],
                condition=condition,
                duration=window_sec,
                base_dir=config.raw_audio_base,
                target_sr_clean=config.target_sr_clean,
                target_sr_phone=config.target_sr_phone,
            )

            feats = extract_features(
                audio, sr,
                feature_type=feature_type,
                aggregate=True,
                n_mfcc=config.n_mfcc,
                n_mels=config.n_mels,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
                fmax=min(sr / 2, config.fmax_clean),
            )

            X_list.append(feats)
            y_list.append(get_label(row, config.label_map))

        except (FileNotFoundError, ValueError) as e:
            # Skip missing or too-short files
            continue

    if not X_list:
        return np.array([]), np.array([])

    return np.stack(X_list), np.array(y_list)


# ============================================================
# PYTORCH DATASET (for CNN)
# ============================================================

class ASVspoofDataset(Dataset):
    """PyTorch Dataset that yields log-Mel spectrograms.

    Each ``__getitem__`` returns ``(spectrogram, label)`` where
    spectrogram has shape ``(1, n_mels, max_frames)``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        condition: str = "clean",
        window_sec: float | None = None,
        n_mels: int = 64,
        max_frames: int = 300,
        config=None,
    ):
        self.df = df.reset_index(drop=True)
        self.condition = condition
        self.window_sec = window_sec
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.config = config or DEFAULT_CONFIG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        audio, sr = load_audio(
            file_name=row["file"],
            split=row["split"],
            condition=self.condition,
            duration=self.window_sec,
            base_dir=self.config.raw_audio_base,
            target_sr_clean=self.config.target_sr_clean,
            target_sr_phone=self.config.target_sr_phone,
        )

        mel = extract_spectrogram_for_cnn(
            audio, sr,
            n_mels=self.n_mels,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            fmax=min(sr / 2, self.config.fmax_clean),
            max_frames=self.max_frames,
        )

        # Add channel dimension: (1, n_mels, max_frames)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)
        label = get_label(row, self.config.label_map)

        return mel_tensor, label


# ============================================================
# COLLATE FOR CNN
# ============================================================

def collate_cnn(batch: List[Tuple[torch.Tensor, int]]):
    """Default collate: stack spectrograms and labels."""
    specs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return specs, labels
