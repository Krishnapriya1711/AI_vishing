#!/usr/bin/env python3
"""Entry point: sliding-window detection experiment.

Usage::

    python experiments/run_sliding_window.py --window 2.0 --hop 1.0 \
        --feature mfcc --max-eval 1000

The script first trains a baseline model on the training split (clean
condition), then runs sliding-window detection on the eval split and
reports aggregated results.  A latency analysis is also produced.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config
from src.data.dataset import load_metadata, filter_metadata, prepare_flat_features
from src.models.baselines import BaselineModel
from src.analysis.sliding_window import (
    run_sliding_window_experiment,
    analyze_detection_latency,
)


def main():
    parser = argparse.ArgumentParser(
        description="Sliding-window detection experiment",
    )
    parser.add_argument(
        "--feature", default="mfcc",
        choices=["mfcc", "mel", "pitch"],
        help="Feature type (default: mfcc)",
    )
    parser.add_argument(
        "--window", type=float, default=2.0,
        help="Window size in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--hop", type=float, default=None,
        help="Hop size in seconds (default: window/2)",
    )
    parser.add_argument(
        "--condition", default="clean",
        choices=["clean", "telephone", "telephone_codec"],
        help="Audio condition (default: clean)",
    )
    parser.add_argument(
        "--aggregation", default="average",
        choices=["average", "majority", "max"],
        help="Score aggregation strategy (default: average)",
    )
    parser.add_argument(
        "--max-train", type=int, default=5000,
        help="Max training samples (default: 5000)",
    )
    parser.add_argument(
        "--max-eval", type=int, default=1000,
        help="Max eval files (default: 1000)",
    )
    parser.add_argument(
        "--model-type", default="random_forest",
        choices=["logistic_regression", "random_forest", "svm"],
        help="Baseline model type (default: random_forest)",
    )
    parser.add_argument(
        "--output", default="outputs/sliding_window_results.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    config = Config()
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SLIDING-WINDOW DETECTION EXPERIMENT")
    print(f"  Window: {args.window}s, Hop: {args.hop or args.window/2:.2f}s")
    print(f"  Feature: {args.feature}, Model: {args.model_type}")
    print(f"  Condition: {args.condition}, Aggregation: {args.aggregation}")
    print(f"  Max train: {args.max_train}, Max eval: {args.max_eval}")
    print("=" * 70)

    # --- Train a baseline model ---
    print("\n--- Training baseline model ---")
    stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)
    train_df = stats_df[stats_df["split"] == "train"]
    if args.max_train:
        train_df = filter_metadata(train_df, max_samples=args.max_train)

    X_train, y_train = prepare_flat_features(
        train_df, condition=args.condition,
        feature_type=args.feature, config=config,
    )

    if len(X_train) == 0:
        print("ERROR: No training features extracted. Audio data may be missing.")
        return

    model = BaselineModel(args.model_type)
    model.fit(X_train, y_train)
    print(f"  Trained {args.model_type} on {len(X_train)} samples")

    # --- Sliding-window detection ---
    results = run_sliding_window_experiment(
        model=model,
        feature_type=args.feature,
        window_sec=args.window,
        hop_sec=args.hop,
        condition=args.condition,
        aggregation=args.aggregation,
        config=config,
        max_eval=args.max_eval,
        save_path=args.output,
    )

    # --- Latency analysis ---
    print("\n--- Latency analysis ---")
    latency = analyze_detection_latency(
        model=model,
        feature_type=args.feature,
        window_sec=args.window,
        hop_sec=args.hop,
        condition=args.condition,
        config=config,
        max_eval=min(args.max_eval, 200),
    )

    if not latency.empty:
        latency_path = Path(args.output).parent / "sliding_window_latency.csv"
        latency.to_csv(latency_path, index=False)
        print(f"  Mean decision time: {latency['decision_sec'].mean():.2f}s")
        print(f"  Median decision time: {latency['decision_sec'].median():.2f}s")
        print(f"  Latency saved to {latency_path}")


if __name__ == "__main__":
    main()
