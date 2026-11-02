"""Analysis subpackage."""

from .sliding_window import (
    slide_windows,
    aggregate_window_scores,
    SlidingWindowDetector,
    run_sliding_window_experiment,
    analyze_detection_latency,
)
