"""Phone / codec robustness experiment.

Tests how telephone channel degradation and G.711 A-law codec
compression affect detection performance.

Two protocols:

1. **Matched condition**: train and evaluate under the same condition.
2. **Cross condition**: train on clean, evaluate on each condition
   (clean, telephone, telephone_codec).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.config import DEFAULT_CONFIG
from ..data.dataset import (
    load_metadata,
    filter_metadata,
    prepare_flat_features,
)
from ..models.baselines import train_baselines, evaluate_baselines
from ..evaluation.metrics import format_results


# ============================================================
# CONDITIONS
# ============================================================

CONDITIONS = ["clean", "telephone", "telephone_codec"]


# ============================================================
# MATCHED-CONDITION EXPERIMENT
# ============================================================

def run_matched_condition_baselines(
    feature_type: str = "mfcc",
    window_sec: float | None = None,
    config=None,
    max_train: int | None = 5000,
    max_eval: int | None = 5000,
) -> Dict[str, pd.DataFrame]:
    """Train and evaluate baselines under each matched condition.

    Returns
    -------
    dict  condition → results DataFrame (one row per model)
    """
    if config is None:
        config = DEFAULT_CONFIG

    all_results = {}

    for condition in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"MATCHED CONDITION: {condition.upper()}")
        print(f"{'='*60}")

        try:
            stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)

            train_df = stats_df[stats_df["split"] == "train"]
            eval_df = stats_df[stats_df["split"] == "eval"]

            if max_train:
                train_df = filter_metadata(train_df, max_samples=max_train)
            if max_eval:
                eval_df = filter_metadata(eval_df, max_samples=max_eval)

            print(f"  Train: {len(train_df)}, Eval: {len(eval_df)}")

            X_train, y_train = prepare_flat_features(
                train_df, condition=condition, window_sec=window_sec,
                feature_type=feature_type, config=config,
            )
            X_eval, y_eval = prepare_flat_features(
                eval_df, condition=condition, window_sec=window_sec,
                feature_type=feature_type, config=config,
            )

            if len(X_train) == 0 or len(X_eval) == 0:
                print("  WARNING: No features extracted.")
                all_results[condition] = pd.DataFrame()
                continue

            models = train_baselines(X_train, y_train)
            results = evaluate_baselines(models, X_eval, y_eval)

            rows = []
            for model_name, res in results.items():
                rows.append({
                    "condition": condition,
                    "model": model_name,
                    "eer": res["eer"],
                    "min_tdcf": res["min_tdcf"],
                    "accuracy": res["accuracy"],
                    "f1": res["f1"],
                })
            all_results[condition] = pd.DataFrame(rows)

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[condition] = pd.DataFrame()

    return all_results


# ============================================================
# CROSS-CONDITION EXPERIMENT
# ============================================================

def run_cross_condition_baselines(
    feature_type: str = "mfcc",
    window_sec: float | None = None,
    config=None,
    max_train: int | None = 5000,
    max_eval: int | None = 5000,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Train on clean, evaluate on all conditions.

    Returns
    -------
    DataFrame  one row per (eval_condition, model) with metrics.
    """
    if config is None:
        config = DEFAULT_CONFIG

    stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)

    train_df = stats_df[stats_df["split"] == "train"]
    eval_df = stats_df[stats_df["split"] == "eval"]

    if max_train:
        train_df = filter_metadata(train_df, max_samples=max_train)
    if max_eval:
        eval_df = filter_metadata(eval_df, max_samples=max_eval)

    # Train on clean
    print(f"\n{'='*60}")
    print("TRAINING ON CLEAN CONDITION")
    print(f"{'='*60}")

    X_train, y_train = prepare_flat_features(
        train_df, condition="clean", window_sec=window_sec,
        feature_type=feature_type, config=config,
    )

    if len(X_train) == 0:
        print("  ERROR: No training features extracted.")
        return pd.DataFrame()

    models = train_baselines(X_train, y_train)

    # Evaluate on each condition
    all_rows = []

    for eval_condition in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"EVALUATING ON: {eval_condition.upper()}")
        print(f"{'='*60}")

        X_eval, y_eval = prepare_flat_features(
            eval_df, condition=eval_condition, window_sec=window_sec,
            feature_type=feature_type, config=config,
        )

        if len(X_eval) == 0:
            print("  WARNING: No eval features.")
            continue

        results = evaluate_baselines(models, X_eval, y_eval)

        for model_name, res in results.items():
            all_rows.append({
                "train_condition": "clean",
                "eval_condition": eval_condition,
                "model": model_name,
                "eer": res["eer"],
                "min_tdcf": res["min_tdcf"],
                "accuracy": res["accuracy"],
                "f1": res["f1"],
            })

    results_df = pd.DataFrame(all_rows)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")

    if not results_df.empty:
        print("\n" + "=" * 70)
        print("CROSS-CONDITION ROBUSTNESS SUMMARY")
        print("=" * 70)
        pivot = results_df.pivot_table(
            values="eer", index="eval_condition",
            columns="model", aggfunc="mean",
        )
        print(pivot.to_string())

    return results_df


# ============================================================
# FULL ROBUSTNESS EXPERIMENT
# ============================================================

def run_phone_robustness_experiment(
    feature_type: str = "mfcc",
    window_sec: float | None = None,
    config=None,
    max_train: int | None = 5000,
    max_eval: int | None = 5000,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Run both matched-condition and cross-condition experiments.

    Returns
    -------
    DataFrame  all results combined.
    """
    print("\n" + "#" * 70)
    print("# MATCHED-CONDITION EXPERIMENTS")
    print("#" * 70)

    matched = run_matched_condition_baselines(
        feature_type=feature_type, window_sec=window_sec,
        config=config, max_train=max_train, max_eval=max_eval,
    )

    print("\n" + "#" * 70)
    print("# CROSS-CONDITION EXPERIMENTS")
    print("#" * 70)

    cross = run_cross_condition_baselines(
        feature_type=feature_type, window_sec=window_sec,
        config=config, max_train=max_train, max_eval=max_eval,
        save_path=save_path,
    )

    # Combine
    matched_rows = []
    for condition, df in matched.items():
        matched_rows.append(df)
    matched_df = pd.concat(matched_rows, ignore_index=True) if matched_rows else pd.DataFrame()

    return pd.concat([matched_df, cross], ignore_index=True)
