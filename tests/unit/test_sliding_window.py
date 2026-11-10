"""Tests for sliding-window detection."""

import numpy as np
import pytest

from src.analysis.sliding_window import (
    slide_windows,
    aggregate_window_scores,
    SlidingWindowDetector,
)
from src.models.baselines import BaselineModel
from src.features.extractors import extract_features
from tests.conftest import generate_bonafide_audio, generate_spoof_audio


# ============================================================
# FIXTURE: model trained on real MFCC features
# ============================================================

@pytest.fixture
def mfcc_trained_model():
    """Train a baseline model on MFCC features extracted from synthetic
    audio so the feature dimensionality matches what the sliding-window
    detector will produce.
    """
    X_list = []
    y_list = []

    # 10 bonafide + 10 spoof
    for i in range(10):
        audio = generate_bonafide_audio(duration=2.0, sr=16000, seed=i)
        feats = extract_features(audio, 16000, "mfcc", aggregate=True)
        X_list.append(feats)
        y_list.append(0)

    for i in range(10):
        audio = generate_spoof_audio(duration=2.0, sr=16000, seed=100 + i)
        feats = extract_features(audio, 16000, "mfcc", aggregate=True)
        X_list.append(feats)
        y_list.append(1)

    X = np.stack(X_list)
    y = np.array(y_list)

    model = BaselineModel("logistic_regression")
    model.fit(X, y)
    return model


# ============================================================
# SLIDE WINDOWS TESTS
# ============================================================

class TestSlideWindows:
    def test_basic_slide(self):
        audio = np.ones(16000 * 10)  # 10 seconds at 16 kHz
        windows = slide_windows(audio, 16000, window_sec=2.0, hop_sec=1.0)
        # 10s audio, 2s window, 1s hop: windows at 0-2, 1-3, ..., 8-10
        # Positions: 0, 1, 2, 3, 4, 5, 6, 7, 8 → 9 windows
        assert len(windows) == 9
        assert all(w.shape[0] == 32000 for w in windows)  # 2 * 16000

    def test_default_hop(self):
        audio = np.ones(16000 * 10)
        windows = slide_windows(audio, 16000, window_sec=2.0)
        # 50% overlap: hop = 1s, same as above
        assert len(windows) == 9

    def test_short_audio(self):
        """Audio shorter than window should produce one window."""
        audio = np.ones(16000 * 1)  # 1 second
        windows = slide_windows(audio, 16000, window_sec=2.0)
        assert len(windows) == 1

    def test_exact_fit(self):
        """Audio exactly matching one window."""
        audio = np.ones(16000 * 2)
        windows = slide_windows(audio, 16000, window_sec=2.0, hop_sec=1.0)
        # Positions: 0, 1 → but 1+2=3 > 2, so only 0
        assert len(windows) == 1

    def test_empty_audio(self):
        audio = np.array([])
        windows = slide_windows(audio, 16000, window_sec=2.0)
        assert len(windows) == 0


# ============================================================
# AGGREGATION TESTS
# ============================================================

class TestAggregation:
    def test_average(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = aggregate_window_scores(scores, "average")
        assert result == pytest.approx(0.5)

    def test_majority(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        result = aggregate_window_scores(scores, "majority", threshold=0.5)
        assert result == pytest.approx(0.6)  # 3 out of 5 >= 0.5

    def test_max(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = aggregate_window_scores(scores, "max")
        assert result == pytest.approx(0.9)

    def test_empty_scores(self):
        result = aggregate_window_scores(np.array([]), "average")
        assert result == 0.5

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            aggregate_window_scores(np.array([0.5]), "unknown")


# ============================================================
# SLIDING WINDOW DETECTOR TESTS
# ============================================================

class TestSlidingWindowDetector:
    def test_score_windows(self, mfcc_trained_model):
        detector = SlidingWindowDetector(
            model=mfcc_trained_model,
            feature_type="mfcc",
            window_sec=1.0,
            hop_sec=0.5,
            aggregation="average",
        )

        audio = generate_bonafide_audio(duration=3.0, sr=16000, seed=42)
        scores = detector.score_windows(audio, 16000)

        # 3s audio, 1s window, 0.5s hop: positions 0, 0.5, 1, 1.5, 2 → 5 windows
        assert len(scores) == 5
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_score_audio(self, mfcc_trained_model):
        detector = SlidingWindowDetector(
            model=mfcc_trained_model, feature_type="mfcc",
            window_sec=1.0, hop_sec=0.5,
        )
        audio = generate_bonafide_audio(duration=3.0, sr=16000, seed=42)
        score = detector.score_audio(audio, 16000)
        assert 0.0 <= score <= 1.0

    def test_aggregation_strategies(self, mfcc_trained_model):
        audio = generate_bonafide_audio(duration=3.0, sr=16000, seed=42)
        for strategy in ["average", "majority", "max"]:
            detector = SlidingWindowDetector(
                model=mfcc_trained_model, feature_type="mfcc",
                window_sec=1.0, hop_sec=0.5,
                aggregation=strategy,
            )
            score = detector.score_audio(audio, 16000)
            assert 0.0 <= score <= 1.0
