"""Sliding-window detection for real-time streaming scenarios.

Instead of evaluating on a single cropped window, we slide a
fixed-length window across the entire audio file and aggregate
per-window scores into a single per-file decision.

Aggregation strategies:

* ``average`` — mean spoof-probability across all windows
* ``majority`` — fraction of windows classified as spoof, threshold at 0.5
* ``max`` — the maximum spoof-probability across all windows
  (most conservative — flags if *any* window looks suspicious)
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.config import DEFAULT_CONFIG
from ..utils.audio import load_audio
from ..features.extractors import extract_features
from ..evaluation.metrics import evaluate_predictions, format_results


# ============================================================
# SLIDING WINDOW EXTRACTOR
# ============================================================

def slide_windows(
    audio: np.ndarray,
    sr: int,
    window_sec: float,
    hop_sec: float | None = None,
) -> List[np.ndarray]:
    """Split audio into overlapping windows.

    Parameters
    ----------
    audio : np.ndarray
    sr : int
    window_sec : float
        Window length in seconds.
    hop_sec : float or None
        Hop length in seconds.  Defaults to ``window_sec / 2`` (50% overlap).

    Returns
    -------
    list of np.ndarray  (one per window)
    """
    if hop_sec is None:
        hop_sec = window_sec / 2.0

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    windows = []
    start = 0

    while start + window_samples <= len(audio):
        windows.append(audio[start:start + window_samples])
        start += hop_samples

    # If the audio is shorter than one window, use the whole thing
    if not windows and len(audio) > 0:
        windows.append(audio[:window_samples] if len(audio) >= window_samples
                        else np.pad(audio, (0, window_samples - len(audio))))

    return windows


# ============================================================
# AGGREGATION STRATEGIES
# ============================================================

def aggregate_window_scores(
    scores: np.ndarray,
    strategy: str = "average",
    threshold: float = 0.5,
) -> float:
    """Aggregate per-window spoof-probabilities into a single score.

    Parameters
    ----------
    scores : np.ndarray
        Spoof-probabilities, one per window (higher = more likely spoof).
    strategy : str
        ``"average"``, ``"majority"``, or ``"max"``.
    threshold : float
        Decision threshold for majority voting.

    Returns
    -------
    float  aggregated score (higher = more likely spoof)
    """
    if len(scores) == 0:
        return 0.5

    if strategy == "average":
        return float(np.mean(scores))

    elif strategy == "majority":
        return float(np.mean(scores >= threshold))

    elif strategy == "max":
        return float(np.max(scores))

    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")


# ============================================================
# SLIDING WINDOW DETECTOR
# ============================================================

class SlidingWindowDetector:
    """Wrap a trained model with sliding-window inference.

    Usage::

        detector = SlidingWindowDetector(model)
        score = detector.score_file("LA_E_123.flac", "eval")
    """

    def __init__(
        self,
        model,
        feature_type: str = "mfcc",
        window_sec: float = 2.0,
        hop_sec: float | None = None,
        condition: str = "clean",
        aggregation: str = "average",
        config=None,
    ):
        self.model = model
        self.feature_type = feature_type
        self.window_sec = window_sec
        self.hop_sec = hop_sec or window_sec / 2.0
        self.condition = condition
        self.aggregation = aggregation
        self.config = config or DEFAULT_CONFIG

    def score_windows(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return spoof-probability for each sliding window."""
        windows = slide_windows(audio, sr, self.window_sec, self.hop_sec)

        if not windows:
            return np.array([0.5])

        scores = []
        for w in windows:
            feats = extract_features(
                w, sr,
                feature_type=self.feature_type,
                aggregate=True,
                n_mfcc=self.config.n_mfcc,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                fmax=min(sr / 2, self.config.fmax_clean),
            )
            feats = feats.reshape(1, -1)
            score = self.model.predict_scores(feats)
            scores.append(score[0])

        return np.array(scores)

    def score_audio(self, audio: np.ndarray, sr: int) -> float:
        """Return aggregated spoof-probability for the full audio."""
        scores = self.score_windows(audio, sr)
        return aggregate_window_scores(scores, self.aggregation)

    def score_file(self, file_name: str, split: str) -> float:
        """Load a file and return its aggregated spoof-probability."""
        audio, sr = load_audio(
            file_name=file_name,
            split=split,
            condition=self.condition,
            duration=None,
            base_dir=self.config.raw_audio_base,
        )
        return self.score_audio(audio, sr)


# ============================================================
# FULL SLIDING-WINDOW EXPERIMENT
# ============================================================

def run_sliding_window_experiment(
    model,
    feature_type: str = "mfcc",
    window_sec: float = 2.0,
    hop_sec: float | None = None,
    condition: str = "clean",
    aggregation: str = "average",
    config=None,
    max_eval: int | None = 1000,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Evaluate sliding-window detection on the eval split.

    Parameters
    ----------
    model : BaselineModel
        Trained baseline model.
    max_eval : int or None
        Maximum number of eval files to process.

    Returns
    -------
    DataFrame  one row per file with window scores and final decision.
    """
    if config is None:
        config = DEFAULT_CONFIG

    stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)
    eval_df = stats_df[stats_df["split"] == "eval"]

    if max_eval and len(eval_df) > max_eval:
        eval_df = eval_df.sample(n=max_eval, random_state=42).reset_index(drop=True)

    detector = SlidingWindowDetector(
        model=model,
        feature_type=feature_type,
        window_sec=window_sec,
        hop_sec=hop_sec,
        condition=condition,
        aggregation=aggregation,
        config=config,
    )

    all_scores = []
    all_labels = []
    all_files = []
    all_n_windows = []

    print(f"\n{'='*60}")
    print(f"SLIDING WINDOW DETECTION")
    print(f"  Window: {window_sec}s, Hop: {hop_sec or window_sec/2:.2f}s")
    print(f"  Condition: {condition}, Aggregation: {aggregation}")
    print(f"  Files: {len(eval_df)}")
    print(f"{'='*60}")

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Scoring"):
        try:
            audio, sr = load_audio(
                file_name=row["file"],
                split=row["split"],
                condition=condition,
                base_dir=config.raw_audio_base,
            )
            scores = detector.score_windows(audio, sr)
            final_score = aggregate_window_scores(scores, aggregation)

            all_files.append(row["file"])
            all_scores.append(final_score)
            all_labels.append(config.label_map.get(row.get("label", "spoof"), 1))
            all_n_windows.append(len(scores))

        except (FileNotFoundError, ValueError) as e:
            continue

    if not all_scores:
        print("  WARNING: No files scored (audio may be missing).")
        return pd.DataFrame()

    labels = np.array(all_labels)
    scores = np.array(all_scores)

    results = evaluate_predictions(labels, scores)
    print(format_results(results))

    results_df = pd.DataFrame({
        "file": all_files,
        "label": ["spoof" if l == 1 else "bonafide" for l in all_labels],
        "score": scores,
        "n_windows": all_n_windows,
    })

    # Add summary metrics
    results_df.attrs["eer"] = results["eer"]
    results_df.attrs["min_tdcf"] = results["min_tdcf"]
    results_df.attrs["accuracy"] = results["accuracy"]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")

    return results_df


# ============================================================
# LATENCY ANALYSIS
# ============================================================

def analyze_detection_latency(
    model,
    feature_type: str = "mfcc",
    window_sec: float = 2.0,
    hop_sec: float | None = None,
    condition: str = "clean",
    config=None,
    max_eval: int = 200,
) -> pd.DataFrame:
    """Analyze how quickly a spoofing decision can be made.

    For each file, tracks the cumulative score after each window
    and finds the earliest point where the decision matches the
    final aggregated decision.

    Returns
    -------
    DataFrame  with columns: file, final_label, final_score,
               decision_window, decision_sec, n_windows
    """
    if config is None:
        config = DEFAULT_CONFIG

    stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)
    eval_df = stats_df[stats_df["split"] == "eval"]
    eval_df = eval_df.sample(n=min(max_eval, len(eval_df)), random_state=42)

    detector = SlidingWindowDetector(
        model=model, feature_type=feature_type,
        window_sec=window_sec, hop_sec=hop_sec,
        condition=condition, config=config,
    )

    rows = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Latency"):
        try:
            audio, sr = load_audio(
                file_name=row["file"], split=row["split"],
                condition=condition, base_dir=config.raw_audio_base,
            )

            scores = detector.score_windows(audio, sr)
            final_score = aggregate_window_scores(scores, "average")
            final_pred = 1 if final_score >= 0.5 else 0

            # Find earliest window where cumulative average matches final decision
            decision_window = len(scores)
            for i in range(1, len(scores) + 1):
                cumul = np.mean(scores[:i])
                pred = 1 if cumul >= 0.5 else 0
                if pred == final_pred:
                    decision_window = i
                    break

            decision_sec = decision_window * (hop_sec or window_sec / 2)

            rows.append({
                "file": row["file"],
                "label": row.get("label", "spoof"),
                "final_score": final_score,
                "n_windows": len(scores),
                "decision_window": decision_window,
                "decision_sec": decision_sec,
            })

        except (FileNotFoundError, ValueError):
            continue

    return pd.DataFrame(rows)
