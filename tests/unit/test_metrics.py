"""Tests for evaluation metrics (EER, min-tDCF, etc.)."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_eer,
    compute_min_tdcf,
    evaluate_predictions,
    format_results,
    BONAFIDE,
    SPOOF,
)


# ============================================================
# EER TESTS
# ============================================================

class TestEER:
    def test_perfect_separation(self):
        """When all spoofs have higher scores than all bonafides,
        EER should be 0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        eer, threshold = compute_eer(labels, scores)
        assert eer < 0.01  # Should be very close to 0

    def test_known_eer(self, known_labels_scores):
        labels, scores = known_labels_scores
        eer, threshold = compute_eer(labels, scores)
        assert 0.0 <= eer <= 1.0
        # With clean separation, EER should be low
        assert eer < 0.1

    def test_random_scores(self):
        """Random scores should give EER close to 0.5."""
        np.random.seed(42)
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.random.uniform(0, 1, 200)
        eer, threshold = compute_eer(labels, scores)
        assert 0.3 < eer < 0.7

    def test_inverted_scores(self):
        """If spoof scores are LOWER than bonafide scores, EER should
        be high (close to 1)."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.7, 0.8, 0.9, 0.1, 0.2, 0.3])
        eer, threshold = compute_eer(labels, scores)
        assert eer > 0.9

    def test_missing_class_raises(self):
        with pytest.raises(ValueError):
            compute_eer(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))

    def test_threshold_in_range(self, known_labels_scores):
        labels, scores = known_labels_scores
        eer, threshold = compute_eer(labels, scores)
        assert scores.min() <= threshold <= scores.max()


# ============================================================
# MIN-tDCF TESTS
# ============================================================

class TestMinTDCF:
    def test_perfect_separation(self):
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.array([0.1] * 10 + [0.9] * 10)
        tdcf = compute_min_tdcf(labels, scores)
        assert tdcf < 0.05

    def test_random_scores(self):
        np.random.seed(42)
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.random.uniform(0, 1, 200)
        tdcf = compute_min_tdcf(labels, scores)
        assert 0.3 < tdcf < 1.0

    def test_inverted_scores(self):
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.array([0.9] * 10 + [0.1] * 10)
        tdcf = compute_min_tdcf(labels, scores)
        assert tdcf > 0.8

    def test_custom_params(self):
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.array([0.1] * 10 + [0.9] * 10)
        tdcf = compute_min_tdcf(labels, scores, p_spoof=0.1, c_miss=1, c_fa=20)
        assert tdcf < 0.05

    def test_missing_class_raises(self):
        with pytest.raises(ValueError):
            compute_min_tdcf(np.array([1, 1, 1]), np.array([0.5, 0.5, 0.5]))


# ============================================================
# EVALUATE PREDICTIONS TESTS
# ============================================================

class TestEvaluatePredictions:
    def test_returns_all_keys(self, known_labels_scores):
        labels, scores = known_labels_scores
        results = evaluate_predictions(labels, scores)
        expected_keys = {
            "eer", "eer_threshold", "min_tdcf", "accuracy",
            "precision", "recall", "f1", "auc", "confusion_matrix",
        }
        assert set(results.keys()) == expected_keys

    def test_perfect_classification(self):
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.array([0.1] * 10 + [0.9] * 10)
        results = evaluate_predictions(labels, scores)
        assert results["eer"] < 0.01
        assert results["accuracy"] > 0.99
        assert results["auc"] > 0.99

    def test_confusion_matrix_shape(self, known_labels_scores):
        labels, scores = known_labels_scores
        results = evaluate_predictions(labels, scores)
        cm = results["confusion_matrix"]
        assert cm.shape == (2, 2)
        assert cm.sum() == len(labels)

    def test_custom_threshold(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.3, 0.6, 0.4, 0.9])
        results = evaluate_predictions(labels, scores, threshold=0.5)
        # At 0.5: 0.3→bonafide (correct), 0.6→spoof (FP), 0.4→bonafide (FN), 0.9→spoof (correct)
        assert results["confusion_matrix"][0, 1] == 1  # FP
        assert results["confusion_matrix"][1, 0] == 1  # FN


# ============================================================
# FORMAT RESULTS TESTS
# ============================================================

class TestFormatResults:
    def test_format_contains_all_metrics(self, known_labels_scores):
        labels, scores = known_labels_scores
        results = evaluate_predictions(labels, scores)
        text = format_results(results)
        assert "EER" in text
        assert "min-tDCF" in text
        assert "Accuracy" in text
        assert "F1" in text
