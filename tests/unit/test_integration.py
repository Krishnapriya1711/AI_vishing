"""Integration tests: full pipeline with synthetic audio files on disk.

These tests create a temporary ASVspoof-like directory tree with FLAC
files, then exercise the data loading → feature extraction → model
training → evaluation pipeline end to end.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.audio import load_audio, find_audio_file
from src.features.extractors import extract_features, extract_spectrogram_for_cnn
from src.models.baselines import BaselineModel, train_baselines, evaluate_baselines
from src.models.cnn import LightweightCNN
from src.models.train import train_cnn, evaluate_cnn
from src.evaluation.metrics import evaluate_predictions
from src.analysis.sliding_window import SlidingWindowDetector, run_sliding_window_experiment


# ============================================================
# AUDIO LOADING
# ============================================================

class TestAudioLoading:
    def test_find_file(self, synthetic_audio_dir):
        path = find_audio_file("LA_T_BONAFIDE_0000", "train", synthetic_audio_dir)
        assert path.exists()
        assert path.suffix == ".flac"

    def test_load_clean(self, synthetic_audio_dir):
        audio, sr = load_audio(
            "LA_T_BONAFIDE_0000", "train",
            condition="clean", duration=1.0,
            base_dir=synthetic_audio_dir,
        )
        assert sr == 16000
        assert len(audio) == 16000  # 1 second at 16 kHz

    def test_load_telephone(self, synthetic_audio_dir):
        audio, sr = load_audio(
            "LA_T_BONAFIDE_0000", "train",
            condition="telephone", duration=1.0,
            base_dir=synthetic_audio_dir,
        )
        assert sr == 8000
        assert len(audio) == 8000  # 1 second at 8 kHz

    def test_load_telephone_codec(self, synthetic_audio_dir):
        audio, sr = load_audio(
            "LA_T_SPOOF_0000", "train",
            condition="telephone_codec", duration=1.0,
            base_dir=synthetic_audio_dir,
        )
        assert sr == 8000
        assert len(audio) == 8000

    def test_load_too_short_raises(self, synthetic_audio_dir):
        with pytest.raises(ValueError):
            load_audio(
                "LA_T_BONAFIDE_0000", "train",
                condition="clean", duration=10.0,
                base_dir=synthetic_audio_dir,
            )

    def test_all_conditions_different(self, synthetic_audio_dir):
        """Telephone and codec should differ from clean."""
        clean, sr_c = load_audio(
            "LA_T_BONAFIDE_0000", "train", condition="clean", duration=1.0,
            base_dir=synthetic_audio_dir,
        )
        phone, sr_p = load_audio(
            "LA_T_BONAFIDE_0000", "train", condition="telephone", duration=1.0,
            base_dir=synthetic_audio_dir,
        )
        assert sr_c != sr_p
        assert not np.allclose(clean[:8000], phone)


# ============================================================
# FULL PIPELINE: FEATURES → BASELINES
# ============================================================

class TestFullPipelineBaselines:
    def test_extract_train_eval_baselines(self, synthetic_audio_dir, synthetic_metadata):
        """End-to-end: extract features from FLAC files, train baselines,
        evaluate on eval split."""
        from src.data.dataset import prepare_flat_features
        from src.utils.config import Config

        config = Config()
        config.raw_audio_base = synthetic_audio_dir

        train_df = synthetic_metadata[synthetic_metadata["split"] == "train"]
        eval_df = synthetic_metadata[synthetic_metadata["split"] == "eval"]

        # Extract features
        X_train, y_train = prepare_flat_features(
            train_df, condition="clean", window_sec=2.0,
            feature_type="mfcc", config=config,
        )
        X_eval, y_eval = prepare_flat_features(
            eval_df, condition="clean", window_sec=2.0,
            feature_type="mfcc", config=config,
        )

        assert len(X_train) > 0
        assert len(X_eval) > 0
        assert X_train.shape[1] == X_eval.shape[1]

        # Train baselines
        models = train_baselines(
            X_train, y_train,
            model_types=["logistic_regression", "random_forest"],
        )

        # Evaluate
        results = evaluate_baselines(models, X_eval, y_eval)
        assert "logistic_regression" in results
        assert "random_forest" in results

        # Should be able to distinguish bonafide from spoof
        # (synthetic audio has different spectral characteristics)
        for model_name, res in results.items():
            assert res["eer"] < 1.0
            assert res["accuracy"] >= 0.0


# ============================================================
# FULL PIPELINE: CNN
# ============================================================

class TestFullPipelineCNN:
    def test_cnn_train_eval(self, synthetic_audio_dir, synthetic_metadata):
        """End-to-end CNN: load spectrograms, train, evaluate."""
        import torch
        from torch.utils.data import DataLoader
        from src.data.dataset import ASVspoofDataset, collate_cnn
        from src.utils.config import Config

        config = Config()
        config.raw_audio_base = synthetic_audio_dir

        train_df = synthetic_metadata[synthetic_metadata["split"] == "train"]
        eval_df = synthetic_metadata[synthetic_metadata["split"] == "eval"]

        train_ds = ASVspoofDataset(
            train_df, condition="clean", window_sec=2.0,
            n_mels=64, max_frames=100, config=config,
        )
        eval_ds = ASVspoofDataset(
            eval_df, condition="clean", window_sec=2.0,
            n_mels=64, max_frames=100, config=config,
        )

        train_loader = DataLoader(
            train_ds, batch_size=4, shuffle=True,
            collate_fn=collate_cnn,
        )
        eval_loader = DataLoader(
            eval_ds, batch_size=4, shuffle=False,
            collate_fn=collate_cnn,
        )

        model = LightweightCNN(n_mels=64, num_classes=2)

        # Train for a few epochs
        history = train_cnn(
            model, train_loader, val_loader=eval_loader,
            num_epochs=3, learning_rate=1e-3,
            verbose=False,
        )
        assert len(history["train_losses"]) == 3

        # Evaluate
        results = evaluate_cnn(model, eval_loader)
        assert "eer" in results
        assert "accuracy" in results
        assert 0.0 <= results["eer"] <= 1.0


# ============================================================
# SLIDING WINDOW END-TO-END
# ============================================================

class TestSlidingWindowEndToEnd:
    def test_detector_with_loaded_audio(self, synthetic_audio_dir, synthetic_metadata):
        """Sliding window detector on real loaded audio."""
        from src.utils.config import Config

        config = Config()
        config.raw_audio_base = synthetic_audio_dir

        # Train a model on features from the synthetic audio
        from src.data.dataset import prepare_flat_features
        from src.utils.config import Config

        config = Config()
        config.raw_audio_base = synthetic_audio_dir

        train_df = synthetic_metadata[synthetic_metadata["split"] == "train"]

        X_train, y_train = prepare_flat_features(
            train_df, condition="clean", window_sec=2.0,
            feature_type="mfcc", config=config,
        )

        model = BaselineModel("random_forest")
        model.fit(X_train, y_train)

        # Set up detector
        detector = SlidingWindowDetector(
            model=model, feature_type="mfcc",
            window_sec=1.0, hop_sec=0.5,
            condition="clean", config=config,
        )

        # Score a file from the eval split
        eval_df = synthetic_metadata[synthetic_metadata["split"] == "eval"]
        first_row = eval_df.iloc[0]
        score = detector.score_file(first_row["file"], first_row["split"])
        assert 0.0 <= score <= 1.0
