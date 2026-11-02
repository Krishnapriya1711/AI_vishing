"""Tests for feature extraction modules."""

import numpy as np
import pytest

from src.features.extractors import (
    extract_mfcc,
    extract_mel_spectrogram,
    extract_pitch,
    aggregate_features,
    extract_features,
    extract_spectrogram_for_cnn,
)


# ============================================================
# MFCC TESTS
# ============================================================

class TestMFCC:
    def test_mfcc_shape(self, bonafide_audio, sample_rate):
        mfcc = extract_mfcc(bonafide_audio, sample_rate)
        # Should be (n_frames, n_mfcc * 3) with deltas
        assert mfcc.ndim == 2
        assert mfcc.shape[1] == 60  # 20 MFCC + 20 delta + 20 delta-delta

    def test_mfcc_no_deltas(self, bonafide_audio, sample_rate):
        mfcc = extract_mfcc(bonafide_audio, sample_rate, include_deltas=False)
        assert mfcc.ndim == 2
        assert mfcc.shape[1] == 20

    def test_mfcc_finite(self, bonafide_audio, sample_rate):
        mfcc = extract_mfcc(bonafide_audio, sample_rate)
        assert np.all(np.isfinite(mfcc))

    def test_mfcc_different_n(self, bonafide_audio, sample_rate):
        mfcc = extract_mfcc(bonafide_audio, sample_rate, n_mfcc=13, include_deltas=False)
        assert mfcc.shape[1] == 13


# ============================================================
# MEL SPECTROGRAM TESTS
# ============================================================

class TestMelSpectrogram:
    def test_mel_shape(self, bonafide_audio, sample_rate):
        mel = extract_mel_spectrogram(bonafide_audio, sample_rate)
        # Should be (n_mels, n_frames)
        assert mel.ndim == 2
        assert mel.shape[0] == 64  # n_mels

    def test_mel_log_scaled(self, bonafide_audio, sample_rate):
        mel_log = extract_mel_spectrogram(bonafide_audio, sample_rate, log_scale=True)
        mel_lin = extract_mel_spectrogram(bonafide_audio, sample_rate, log_scale=False)
        assert not np.allclose(mel_log, mel_lin)

    def test_mel_finite(self, bonafide_audio, sample_rate):
        mel = extract_mel_spectrogram(bonafide_audio, sample_rate)
        assert np.all(np.isfinite(mel))

    def test_mel_different_n_mels(self, bonafide_audio, sample_rate):
        mel = extract_mel_spectrogram(bonafide_audio, sample_rate, n_mels=128)
        assert mel.shape[0] == 128


# ============================================================
# PITCH TESTS
# ============================================================

class TestPitch:
    def test_pitch_shape(self, bonafide_audio, sample_rate):
        pitch = extract_pitch(bonafide_audio, sample_rate)
        assert pitch.ndim == 2
        assert pitch.shape[1] == 1

    def test_pitch_non_negative(self, bonafide_audio, sample_rate):
        pitch = extract_pitch(bonafide_audio, sample_rate)
        assert np.all(pitch >= 0)

    def test_pitch_finite(self, bonafide_audio, sample_rate):
        pitch = extract_pitch(bonafide_audio, sample_rate)
        assert np.all(np.isfinite(pitch))


# ============================================================
# AGGREGATION TESTS
# ============================================================

class TestAggregation:
    def test_aggregate_shape(self):
        features = np.random.randn(100, 20).astype(np.float32)
        agg = aggregate_features(features)
        assert agg.shape == (40,)  # 2 * 20

    def test_aggregate_values(self):
        features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        agg = aggregate_features(features)
        expected_mean = np.array([3.0, 4.0])
        expected_std = np.array([1.6330, 1.6330])  # population std (ddof=0)
        np.testing.assert_allclose(agg[:2], expected_mean, rtol=1e-3)
        np.testing.assert_allclose(agg[2:], expected_std, rtol=1e-3)

    def test_aggregate_empty(self):
        features = np.zeros((0, 5))
        agg = aggregate_features(features)
        assert agg.shape == (10,)
        assert np.all(agg == 0)


# ============================================================
# UNIFIED EXTRACTION TESTS
# ============================================================

class TestUnifiedExtraction:
    def test_mfcc_aggregated(self, bonafide_audio, sample_rate):
        feats = extract_features(bonafide_audio, sample_rate, "mfcc", aggregate=True)
        assert feats.ndim == 1
        assert feats.shape[0] == 120  # 2 * 60

    def test_mel_aggregated(self, bonafide_audio, sample_rate):
        feats = extract_features(bonafide_audio, sample_rate, "mel", aggregate=True)
        assert feats.ndim == 1
        assert feats.shape[0] == 128  # 2 * 64

    def test_pitch_aggregated(self, bonafide_audio, sample_rate):
        feats = extract_features(bonafide_audio, sample_rate, "pitch", aggregate=True)
        assert feats.ndim == 1
        assert feats.shape[0] == 2  # 2 * 1

    def test_unknown_type_raises(self, bonafide_audio, sample_rate):
        with pytest.raises(ValueError):
            extract_features(bonafide_audio, sample_rate, "invalid")

    def test_bonafide_vs_spoof_differ(self, audio_pair, sample_rate):
        bona, spoof = audio_pair
        bona_feats = extract_features(bona, sample_rate, "mfcc", aggregate=True)
        spoof_feats = extract_features(spoof, sample_rate, "mfcc", aggregate=True)
        assert not np.allclose(bona_feats, spoof_feats)


# ============================================================
# CNN SPECTROGRAM TESTS
# ============================================================

class TestCNNSpectrogram:
    def test_fixed_size_output(self, bonafide_audio, sample_rate):
        mel = extract_spectrogram_for_cnn(
            bonafide_audio, sample_rate,
            n_mels=64, max_frames=300,
        )
        assert mel.shape == (64, 300)

    def test_short_audio_padded(self, sample_rate):
        short_audio = np.random.randn(int(sample_rate * 0.1)).astype(np.float32)
        mel = extract_spectrogram_for_cnn(short_audio, sample_rate, max_frames=100)
        assert mel.shape[1] == 100

    def test_normalised(self, bonafide_audio, sample_rate):
        mel = extract_spectrogram_for_cnn(bonafide_audio, sample_rate, max_frames=100)
        # Should be roughly zero-mean
        assert abs(mel.mean()) < 1.0
