"""Tests for baseline models and CNN."""

import numpy as np
import pytest
import torch

from src.models.baselines import BaselineModel, train_baselines, evaluate_baselines
from src.models.cnn import LightweightCNN


# ============================================================
# BASELINE MODEL TESTS
# ============================================================

class TestBaselineModel:
    def test_init_logistic_regression(self):
        model = BaselineModel("logistic_regression")
        assert model.model_type == "logistic_regression"

    def test_init_random_forest(self):
        model = BaselineModel("random_forest")
        assert model.model_type == "random_forest"

    def test_init_svm(self):
        model = BaselineModel("svm")
        assert model.model_type == "svm"

    def test_init_invalid_type(self):
        with pytest.raises(ValueError):
            BaselineModel("invalid_model")

    def test_fit_predict(self, synthetic_train_data, synthetic_eval_data):
        X_train, y_train = synthetic_train_data
        X_eval, y_eval = synthetic_eval_data

        model = BaselineModel("logistic_regression")
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        assert preds.shape == (len(X_eval),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_scores(self, synthetic_train_data, synthetic_eval_data):
        X_train, y_train = synthetic_train_data
        X_eval, _ = synthetic_eval_data

        model = BaselineModel("logistic_regression")
        model.fit(X_train, y_train)

        scores = model.predict_scores(X_eval)
        assert scores.shape == (len(X_eval),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_learned_separation(self, synthetic_train_data, synthetic_eval_data):
        """Models should learn to separate bonafide from spoof."""
        X_train, y_train = synthetic_train_data
        X_eval, y_eval = synthetic_eval_data

        for model_type in ["logistic_regression", "random_forest"]:
            model = BaselineModel(model_type)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            accuracy = np.mean(preds == y_eval)
            # Synthetic data has clear separation, should achieve high accuracy
            assert accuracy > 0.85, f"{model_type} accuracy too low: {accuracy}"

    def test_save_load(self, synthetic_train_data, synthetic_eval_data, tmp_path):
        X_train, y_train = synthetic_train_data
        X_eval, y_eval = synthetic_eval_data

        model = BaselineModel("logistic_regression")
        model.fit(X_train, y_train)
        scores_before = model.predict_scores(X_eval)

        save_path = str(tmp_path / "model.pkl")
        model.save(save_path)
        loaded = BaselineModel.load(save_path)
        scores_after = loaded.predict_scores(X_eval)

        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)


# ============================================================
# BATCH TRAIN/ EVALUATE TESTS
# ============================================================

class TestBatchTrain:
    def test_train_baselines(self, synthetic_train_data):
        X_train, y_train = synthetic_train_data
        models = train_baselines(X_train, y_train,
                                  model_types=["logistic_regression", "random_forest"])
        assert "logistic_regression" in models
        assert "random_forest" in models

    def test_evaluate_baselines(self, synthetic_train_data, synthetic_eval_data):
        X_train, y_train = synthetic_train_data
        X_eval, y_eval = synthetic_eval_data

        models = train_baselines(X_train, y_train,
                                  model_types=["logistic_regression"])
        results = evaluate_baselines(models, X_eval, y_eval)

        assert "logistic_regression" in results
        res = results["logistic_regression"]
        assert "eer" in res
        assert "accuracy" in res


# ============================================================
# CNN MODEL TESTS
# ============================================================

class TestLightweightCNN:
    def test_forward_output_shape(self):
        model = LightweightCNN(n_mels=64, num_classes=2)
        x = torch.randn(4, 1, 64, 300)
        output = model(x)
        assert output.shape == (4, 2)

    def test_predict_proba(self):
        model = LightweightCNN(n_mels=64)
        x = torch.randn(2, 1, 64, 300)
        proba = model.predict_proba(x)
        assert proba.shape == (2, 2)
        # Probabilities should sum to 1
        assert torch.allclose(proba.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_parameter_count(self):
        model = LightweightCNN(n_mels=64)
        n_params = model.num_parameters()
        # Should be a lightweight model (< 500k params)
        assert n_params > 0
        assert n_params < 500_000

    def test_different_mel_sizes(self):
        model = LightweightCNN(n_mels=128)
        x = torch.randn(2, 1, 128, 200)
        output = model(x)
        assert output.shape == (2, 2)

    def test_gradient_flow(self):
        """Verify gradients flow through the network."""
        model = LightweightCNN(n_mels=64)
        x = torch.randn(2, 1, 64, 300)
        labels = torch.tensor([0, 1])
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model(x), labels)
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_batch_inference(self):
        """Test with different batch sizes."""
        model = LightweightCNN(n_mels=64)
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 1, 64, 300)
            output = model(x)
            assert output.shape == (batch_size, 2)
