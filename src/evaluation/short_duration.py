"""Short-duration experiment.

Evaluates how detection performance degrades as the audio window
shrinks from 3 s down to 0.5 s.  For each window size we train the
baseline classifiers (and optionally the CNN) on a training subset and
evaluate on the eval split.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.config import DEFAULT_CONFIG
from ..data.dataset import (
    load_metadata,
    filter_metadata,
    prepare_flat_features,
    ASVspoofDataset,
    collate_cnn,
)
from ..features.extractors import extract_features
from ..models.baselines import BaselineModel, train_baselines, evaluate_baselines
from ..evaluation.metrics import evaluate_predictions, format_results


# ============================================================
# SINGLE WINDOW EXPERIMENT (baselines only)
# ============================================================

def run_single_window_baselines(
    window_sec: float,
    feature_type: str = "mfcc",
    condition: str = "clean",
    config=None,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> Dict[str, dict]:
    """Train and evaluate all baselines for one window size.

    Returns
    -------
    dict  model_name → results dict
    """
    if config is None:
        config = DEFAULT_CONFIG

    stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)
    sw_df = load_metadata("short_window_metadata.csv", config)

    # Filter to the specific window
    sw_df = sw_df[sw_df["window_sec"] == window_sec]

    train_df = sw_df[sw_df["split"] == "train"]
    eval_df = sw_df[sw_df["split"] == "eval"]

    if max_train:
        train_df = filter_metadata(train_df, max_samples=max_train)
    if max_eval:
        eval_df = filter_metadata(eval_df, max_samples=max_eval)

    print(f"\n--- Window {window_sec}s, {feature_type}, {condition} ---")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Eval  samples: {len(eval_df)}")

    # Extract features
    print("  Extracting train features...")
    X_train, y_train = prepare_flat_features(
        train_df, condition=condition, window_sec=window_sec,
        feature_type=feature_type, config=config,
        max_samples=max_train,
    )

    print("  Extracting eval features...")
    X_eval, y_eval = prepare_flat_features(
        eval_df, condition=condition, window_sec=window_sec,
        feature_type=feature_type, config=config,
        max_samples=max_eval,
    )

    if len(X_train) == 0 or len(X_eval) == 0:
        print("  WARNING: No features extracted (audio files may be missing).")
        return {}

    print(f"  Feature dim: {X_train.shape[1]}")

    # Train baselines
    print("  Training baselines...")
    models = train_baselines(X_train, y_train)

    # Evaluate
    print("  Evaluating...")
    results = evaluate_baselines(models, X_eval, y_eval)

    return results


# ============================================================
# FULL SHORT-DURATION EXPERIMENT
# ============================================================

def run_short_duration_experiment(
    feature_type: str = "mfcc",
    condition: str = "clean",
    window_sizes: List[float] | None = None,
    config=None,
    max_train: int | None = 5000,
    max_eval: int | None = 5000,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Run the short-duration experiment across all window sizes.

    Returns
    -------
    DataFrame  one row per (window_size, model) with metrics.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if window_sizes is None:
        window_sizes = config.window_sizes

    all_results = []

    for window in window_sizes:
        try:
            results = run_single_window_baselines(
                window_sec=window,
                feature_type=feature_type,
                condition=condition,
                config=config,
                max_train=max_train,
                max_eval=max_eval,
            )
        except Exception as e:
            print(f"  ERROR for window {window}s: {e}")
            results = {}

        for model_name, res in results.items():
            all_results.append({
                "window_sec": window,
                "model": model_name,
                "feature": feature_type,
                "condition": condition,
                "eer": res["eer"],
                "min_tdcf": res["min_tdcf"],
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "auc": res["auc"],
            })

    results_df = pd.DataFrame(all_results)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")

    # Summary table
    if not results_df.empty:
        print("\n" + "=" * 70)
        print("SHORT-DURATION EXPERIMENT SUMMARY")
        print("=" * 70)
        pivot = results_df.pivot_table(
            values="eer", index="window_sec",
            columns="model", aggfunc="mean",
        )
        print(pivot.to_string())

    return results_df


# ============================================================
# CNN SHORT-DURATION EXPERIMENT
# ============================================================

def run_short_duration_cnn(
    window_sizes: List[float] | None = None,
    condition: str = "clean",
    config=None,
    max_train: int | None = 5000,
    max_eval: int | None = 5000,
    num_epochs: int = 30,
    batch_size: int = 64,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Run the short-duration experiment with the CNN.

    Returns
    -------
    DataFrame  one row per window_size with CNN metrics.
    """
    if config is None:
        config = DEFAULT_CONFIG

    from torch.utils.data import DataLoader
    from ..models.cnn import LightweightCNN
    from ..models.train import train_cnn, evaluate_cnn

    if window_sizes is None:
        window_sizes = config.window_sizes

    all_results = []

    for window in window_sizes:
        print(f"\n{'='*60}")
        print(f"CNN — Window {window}s, {condition}")
        print(f"{'='*60}")

        try:
            sw_df = load_metadata("short_window_metadata.csv", config)
            sw_df = sw_df[sw_df["window_sec"] == window]

            train_df = sw_df[sw_df["split"] == "train"]
            eval_df = sw_df[sw_df["split"] == "eval"]

            if max_train:
                train_df = filter_metadata(train_df, max_samples=max_train)
            if max_eval:
                eval_df = filter_metadata(eval_df, max_samples=max_eval)

            train_ds = ASVspoofDataset(
                train_df, condition=condition, window_sec=window,
                n_mels=config.n_mels, config=config,
            )
            eval_ds = ASVspoofDataset(
                eval_df, condition=condition, window_sec=window,
                n_mels=config.n_mels, config=config,
            )

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                collate_fn=collate_cnn, num_workers=0,
            )
            eval_loader = DataLoader(
                eval_ds, batch_size=batch_size, shuffle=False,
                collate_fn=collate_cnn, num_workers=0,
            )

            model = LightweightCNN(n_mels=config.n_mels)
            print(f"  Parameters: {model.num_parameters():,}")

            train_cnn(
                model, train_loader,
                num_epochs=num_epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                early_stopping_patience=config.early_stopping_patience,
            )

            results = evaluate_cnn(model, eval_loader)
            print(format_results(results))

            all_results.append({
                "window_sec": window,
                "model": "cnn",
                "condition": condition,
                "eer": results["eer"],
                "min_tdcf": results["min_tdcf"],
                "accuracy": results["accuracy"],
                "f1": results["f1"],
                "auc": results["auc"],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(all_results)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")

    return results_df
