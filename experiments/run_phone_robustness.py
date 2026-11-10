#!/usr/bin/env python3
"""Entry point: phone / codec robustness experiment.

Usage::

    python experiments/run_phone_robustness.py --feature mfcc --max-train 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config
from src.evaluation.phone_robustness import run_phone_robustness_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Phone / codec robustness experiment",
    )
    parser.add_argument(
        "--feature", default="mfcc",
        choices=["mfcc", "mel", "pitch"],
        help="Feature type (default: mfcc)",
    )
    parser.add_argument(
        "--window", type=float, default=None,
        help="Window size in seconds (default: full audio)",
    )
    parser.add_argument(
        "--max-train", type=int, default=5000,
        help="Max training samples (default: 5000)",
    )
    parser.add_argument(
        "--max-eval", type=int, default=5000,
        help="Max eval samples (default: 5000)",
    )
    parser.add_argument(
        "--output", default="outputs/phone_robustness_results.csv",
        help="Output CSV path (default: outputs/phone_robustness_results.csv)",
    )

    args = parser.parse_args()

    config = Config()
    config.max_train_samples = args.max_train
    config.max_eval_samples = args.max_eval

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHONE / CODEC ROBUSTNESS EXPERIMENT")
    print(f"  Feature: {args.feature}, Window: {args.window or 'full'}")
    print(f"  Max train: {args.max_train}, Max eval: {args.max_eval}")
    print("=" * 70)

    results = run_phone_robustness_experiment(
        feature_type=args.feature,
        window_sec=args.window,
        config=config,
        max_train=args.max_train,
        max_eval=args.max_eval,
        save_path=args.output,
    )

    if not results.empty:
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
