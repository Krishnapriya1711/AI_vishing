"""Evaluation subpackage."""

from .metrics import (
    compute_eer,
    compute_min_tdcf,
    evaluate_predictions,
    format_results,
)
from .short_duration import run_short_duration_experiment, run_short_duration_cnn
from .phone_robustness import run_phone_robustness_experiment
