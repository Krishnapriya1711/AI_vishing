"""Central configuration for the AI voice vishing detection project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ============================================================
# PROJECT ROOT
# ============================================================

# Resolve relative to this file so imports work from any CWD
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CONFIGURATION DATACLASS
# ============================================================

@dataclass
class Config:
    """All tunable hyperparameters and paths in one place."""

    # --- Paths ---
    metadata_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "metadata"
    )
    output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs"
    )

    # Audio base directory (where ASVspoof raw FLAC files live)
    raw_audio_base: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "LA" / "LA"
    )

    # --- Audio conditions ---
    target_sr_clean: int = 16000
    target_sr_phone: int = 8000
    telephone_low: float = 300.0   # Hz
    telephone_high: float = 3400.0  # Hz

    # --- Short-window durations (seconds) ---
    window_sizes: List[float] = field(
        default_factory=lambda: [3.0, 2.0, 1.5, 1.0, 0.75, 0.5]
    )

    # --- Feature extraction ---
    n_mfcc: int = 20
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160   # 10 ms at 16 kHz, 12.5 ms at 8 kHz
    win_length: int = 400   # 25 ms at 16 kHz
    fmin: float = 0.0
    fmax_clean: float = 8000.0
    fmax_phone: float = 3400.0

    # --- CNN training ---
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4

    # --- Experiment sampling ---
    # Limits for how many files to process (full dataset = 121k files)
    max_train_samples: int = 5000
    max_eval_samples: int = 5000

    # --- ASVspoof tDCF cost parameters ---
    p_spoof: float = 0.05
    c_miss: float = 1.0
    c_fa: float = 10.0

    # --- Label mapping ---
    # 0 = bonafide, 1 = spoof
    label_map: dict = field(
        default_factory=lambda: {"bonafide": 0, "spoof": 1}
    )

    def get_config(self) -> dict:
        """Return a plain dict copy for logging."""
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()}


# Default singleton
DEFAULT_CONFIG = Config()
