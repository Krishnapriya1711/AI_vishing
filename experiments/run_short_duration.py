#!/usr/bin/env python3
"""Entry point: short-duration experiment.

Usage::

    python experiments/run_short_duration.py --feature mfcc --max-train 5000

Run ``python experiments/run_short_duration.py --help`` for all options.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config
from src.evaluation.short_duration import (
    run_short_duration_experiment,
    run_short_duration_cnn,
)


def main():
    parser = argparse.ArgumentParser(
        description="Short-duration detection experiment",
    )
    parser.add_argument(
        "--feature", default="mfcc",
        choices=["mfcc", "mel", "pitch"],
        help="Feature type (default: mfcc)",
    )
    parser.add_argument(
        "--condition", default="clean",
        choices=["clean", "telephone", "telephone_codec"],
        help="Audio condition (default: clean)",
    )
    parser.add_argument(
        "--max-train", type=int, default=5000,
        help="Max training samples per window (default: 5000)",
    )
    parser.add_argument(
        "--max-eval", type=int, default=5000,
        help="Max eval samples per window (default: 5000)",
    )
    parser.add_argument(
        "--cnn", action="store_true",
        help="Also run the CNN model",
    )
    parser.add_argument(
        "--cnn-epochs", type=int, default=30,
        help="CNN training epochs (default: 30)",
    )
    parser.add_argument(
        "--output", default="outputs/short_duration_results.csv",
        help="Output CSV path (default: outputs/short_duration_results.csv)",
    )
    parser.add_argument(
        "--windows", nargs="*", type=float,
        default=None,
        help="Window sizes to test (default: all)",
    )

    args = parser.parse_args()

    config = Config()
    config.max_train_samples = args.max_train
    config.max_eval_samples = args.max_eval

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SHORT-DURATION EXPERIMENT")
    print(f"  Feature: {args.feature}, Condition: {args.condition}")
    print(f"  Windows: {args.windows or config.window_sizes}")
    print(f"  Max train: {args.max_train}, Max eval: {args.max_eval}")
    print("=" * 70)

    # Baselines
    baseline_results = run_short_duration_experiment(
        feature_type=args.feature,
        condition=args.condition,
        window_sizes=args.windows,
        config=config,
        max_train=args.max_train,
        max_eval=args.max_eval,
        save_path=args.output,
    )

    # CNN
    if args.cnn:
        cnn_output = args.output.replace(".csv", "_cnn.csv")
        cnn_results = run_short_duration_cnn(
            window_sizes=args.windows,
            condition=args.condition,
            config=config,
            max_train=args.max_train,
            max_eval=args.max_eval,
            num_epochs=args.cnn_epochs,
            save_path=cnn_output,
        )


if __name__ == "__main__":
    main()
