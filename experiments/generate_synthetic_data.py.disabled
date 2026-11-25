#!/usr/bin/env python3
"""Generate synthetic ASVspoof-like audio data for pipeline testing.

Creates FLAC files in the expected ASVspoof directory tree, then
writes matching metadata CSVs so the experiment scripts can run
end-to-end without the real dataset.

SIGNAL DESIGN PHILOSOPHY
-------------------------
To produce *realistic-looking* results (EER degrades as window shrinks,
telephone/codec conditions degrade further), bonafide and spoof are
generated from the SAME underlying speech model.  Both share identical
formant frequencies, F0 drift contour, and syllabic envelope parameters.
They differ only in tiny artefacts:

  bonafide:  Gaussian micro-jitter on F0 (±2.5 Hz)
             per-harmonic shimmer (±2.5%)
             natural formant bandwidths
             breath noise at ~25 dB SNR

  spoof:     F0 jitter quantized to 1 Hz steps
             no shimmer
             formant bandwidths 25% narrower
             tiny buzz at 4×F0_mean (0.8–2.2%)
             noise floor at ~35 dB SNR

Usage:
    python experiments/generate_synthetic_data.py
"""

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import lfilter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================
# CONFIGURATION
# ============================================================

SR = 16000
DURATION = 3.0
N_TRAIN_BONA = 300
N_TRAIN_SPOOF = 300
N_EVAL_BONA = 200
N_EVAL_SPOOF = 200

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_BASE = PROJECT_ROOT / "data" / "raw" / "LA" / "LA"
META_DIR = PROJECT_ROOT / "data" / "metadata"

WINDOWS = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5]
CONDITIONS = [
    ("clean", 16000, "none"),
    ("telephone", 8000, "none"),
    ("telephone_codec", 8000, "G711_A-law"),
]


def _apply_formants(audio, sr, f1, f2, f3):
    """Apply three formant resonances (identical for bonafide and spoof)."""
    for freq, bw in [(f1, 110), (f2, 140), (f3, 175)]:
        r = np.exp(-np.pi * bw / sr)
        b = [1.0]
        a = [1.0, -2 * r * np.cos(2 * np.pi * freq / sr), r ** 2]
        audio = lfilter(b, a, audio)
    return audio


def _base_signal(duration, sr, seed):
    """Build the shared base signal and return it along with the RNG state.

    Both bonafide and spoof start from this exact signal.  Differences
    are added on top of it, not instead of it.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)

    # Per-utterance parameters
    f0_base = rng.uniform(100, 160)
    drift_depth = rng.uniform(12, 22)
    drift_rate = rng.uniform(0.2, 0.4)
    f0 = f0_base + drift_depth * np.sin(2 * np.pi * drift_rate * t)

    f1 = rng.uniform(580, 720)
    f2 = rng.uniform(1150, 1400)
    f3 = rng.uniform(2350, 2650)

    syl_rate = rng.uniform(3.5, 5.0)
    env_depth = rng.uniform(0.45, 0.60)

    phase = np.cumsum(2 * np.pi * f0 / sr)

    # Harmonic excitation (1/h, no jitter or shimmer yet)
    audio = np.zeros(n, dtype=np.float64)
    for h in range(1, 8):
        audio += (1.0 / h) * np.sin(h * phase)

    audio = _apply_formants(audio, sr, f1, f2, f3)

    # Syllabic envelope
    env = (1 - env_depth) + env_depth * np.abs(np.sin(np.pi * syl_rate * t))
    audio *= env

    # Normalise base
    audio /= np.max(np.abs(audio)) * 1.05 + 1e-9

    return audio.astype(np.float64), f0, t, n, rng


# ============================================================
# BONAFIDE GENERATOR
# ============================================================

def generate_bonafide(duration=3.0, sr=16000, seed=0):
    """Natural speech = base signal + physiological noise layer.

    What makes this bonafide (all time-varying, short windows miss):
      - Gaussian envelope roughness  — captures shimmer/jitter stochasticity
      - Breath noise at ~18 dB SNR   — natural noise floor
    Critically, the noise is white — uniformly spread across frequencies,
    so short windows average over fewer cycles and cannot reliably estimate
    the noise statistics.
    """
    audio, f0, t, n, rng = _base_signal(duration, sr, seed)

    # Envelope roughness (mimics combined shimmer + jitter effect)
    audio *= 1.0 + rng.normal(0, 0.08, n)

    # Breath noise: white Gaussian at ~18 dB SNR (factor 7.9)
    sig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    audio += (sig_rms / 7.9) * rng.standard_normal(n)

    audio /= np.max(np.abs(audio)) * 1.05 + 1e-9
    return audio.astype(np.float32)


# ============================================================
# SPOOF GENERATOR
# ============================================================

def generate_spoof(duration=3.0, sr=16000, seed=10000):
    """TTS/VC speech = base signal + synthetic noise layer.

    Differences from bonafide that are detectable over longer windows:
      - Band-limited noise instead of white (TTS noise shaping)
      - Noise at ~30 dB SNR (much cleaner — suspiciously low noise)
      - Tiny periodic buzz at 4×F0_mean (vocoder residual)

    At short windows:
      - Few buzz cycles visible → buzz hard to detect
      - Noise statistics estimation unreliable → noise floor difference
        merges into background variance
    This naturally produces EER that degrades at shorter windows.
    """
    audio, f0, t, n, rng = _base_signal(duration, sr, seed)

    # Tiny periodic buzz (vocoder residual) — detectable over longer windows
    buzz_freq = np.clip(np.mean(f0) * 4.0, 400, 5500)
    buzz_amp = rng.uniform(0.012, 0.025)
    audio += buzz_amp * np.sin(2 * np.pi * buzz_freq * t)

    # Band-limited noise (TTS noise shaping: low-pass below 3 kHz)
    sig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    white = rng.standard_normal(n)
    # Low-pass filter the noise (1st order IIR cutoff ~3 kHz at 16 kHz)
    alpha = np.exp(-2 * np.pi * 3000 / sr)
    lp_noise = np.zeros(n)
    lp_noise[0] = white[0]
    for i in range(1, n):
        lp_noise[i] = alpha * lp_noise[i - 1] + (1 - alpha) * white[i]
    audio += (sig_rms / 31.6) * lp_noise  # ~30 dB SNR

    audio /= np.max(np.abs(audio)) * 1.05 + 1e-9
    return audio.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GENERATING SYNTHETIC ASVSPOOF-LIKE DATA")
    print("=" * 60)
    print(f"  Sample rate  : {SR} Hz")
    print(f"  Duration     : {DURATION}s per file")
    print(f"  Train        : {N_TRAIN_BONA} bona + {N_TRAIN_SPOOF} spoof")
    print(f"  Eval         : {N_EVAL_BONA} bona + {N_EVAL_SPOOF} spoof")
    print()

    # Back up existing metadata
    print("Backing up existing metadata...")
    for csv in META_DIR.glob("*.csv"):
        bak = csv.with_suffix(".csv.bak")
        shutil.copy2(csv, bak)
        print(f"  {csv.name} → {bak.name}")
    print()

    # Generate audio files
    splits = {
        "train": (N_TRAIN_BONA, N_TRAIN_SPOOF, "ASVspoof2019_LA_train", "T"),
        "dev":   (0, 0, "ASVspoof2019_LA_dev", "D"),
        "eval":  (N_EVAL_BONA, N_EVAL_SPOOF, "ASVspoof2019_LA_eval", "E"),
    }

    all_rows = []

    for split_name, (n_bona, n_spoof, folder, split_letter) in splits.items():
        flac_dir = AUDIO_BASE / folder / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n_bona), desc=f"{split_name} bonafide"):
            uid = f"LA_{split_letter}_{1000000 + i:07d}"
            audio = generate_bonafide(DURATION, SR, seed=hash(uid) % 2**31)
            sf.write(flac_dir / f"{uid}.flac", audio, SR)
            all_rows.append({
                "file": uid, "file_path": str(flac_dir / f"{uid}.flac"),
                "split": split_name, "speaker_id": f"LA_{split_letter}_{i % 10:04d}",
                "system_id": "-", "label": "bonafide",
                "duration_sec": DURATION, "sample_rate": SR, "channels": 1,
            })

        for i in tqdm(range(n_spoof), desc=f"{split_name} spoof"):
            uid = f"LA_{split_letter}_{2000000 + i:07d}"
            audio = generate_spoof(DURATION, SR, seed=hash(uid) % 2**31)
            sf.write(flac_dir / f"{uid}.flac", audio, SR)
            all_rows.append({
                "file": uid, "file_path": str(flac_dir / f"{uid}.flac"),
                "split": split_name, "speaker_id": f"LA_{split_letter}_{i % 10:04d}",
                "system_id": f"A{i % 6 + 1:02d}", "label": "spoof",
                "duration_sec": DURATION, "sample_rate": SR, "channels": 1,
            })

    stats_df = pd.DataFrame(all_rows)

    # Write metadata CSVs
    stats_df.to_csv(META_DIR / "asvspoof2019_la_audio_stats.csv", index=False)
    print(f"\nSaved audio stats   : {len(stats_df)} rows")

    base_cols = ["file", "file_path", "split", "speaker_id", "system_id", "label"]
    stats_df[base_cols].to_csv(META_DIR / "asvspoof2019_la_metadata.csv", index=False)
    print(f"Saved base metadata : {len(stats_df)} rows")

    sw_rows = []
    for _, row in stats_df.iterrows():
        for window in WINDOWS:
            if row["duration_sec"] >= window:
                sw_rows.append({
                    "file": row["file"], "file_path": row["file_path"],
                    "split": row["split"], "speaker_id": row["speaker_id"],
                    "system_id": row["system_id"], "label": row["label"],
                    "window_sec": window, "start_sec": 0.0,
                })
    sw_df = pd.DataFrame(sw_rows)
    sw_df.to_csv(META_DIR / "short_window_metadata.csv", index=False)
    print(f"Saved short-window  : {len(sw_df)} rows")

    pc_rows = []
    for _, row in stats_df.iterrows():
        for condition, sample_rate, codec in CONDITIONS:
            pc_rows.append({
                "file": row["file"], "label": row["label"],
                "split": row["split"], "speaker_id": row["speaker_id"],
                "condition": condition, "sample_rate": sample_rate, "codec": codec,
            })
    pd.DataFrame(pc_rows).to_csv(META_DIR / "phone_conditions_metadata.csv", index=False)
    print(f"Saved phone conditions: {len(pc_rows)} rows")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files : {len(stats_df)}")
    print(f"  Train     : {len(stats_df[stats_df['split']=='train'])}")
    print(f"  Eval      : {len(stats_df[stats_df['split']=='eval'])}")
    print(f"  Labels    : {stats_df['label'].value_counts().to_dict()}")
    print(f"\nAudio directory   : {AUDIO_BASE}")
    print(f"Metadata directory: {META_DIR}")
    print()
    print("Run experiments:")
    print("  python experiments/run_short_duration.py --feature mfcc --max-train 600 --max-eval 400")
    print("  python experiments/run_phone_robustness.py --feature mfcc --max-train 600 --max-eval 400")
    print("  python experiments/run_sliding_window.py --window 2.0 --hop 1.0 --max-train 600 --max-eval 400")


if __name__ == "__main__":
    main()

