"""
Unit tests for FGSM perturbation generator.
Tests gradient computation, epsilon bounds, MSE constraints, and batch mode.
"""

import numpy as np
import pytest
import torch

from src.core.config import load_config
from src.models.model_zoo import ModelZoo
from src.perturbation.fgsm import FGSMPerturbation


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def model_zoo(config):
    zoo = ModelZoo(config)
    zoo.preload_all()
    return zoo


@pytest.fixture
def fgsm(config):
    return FGSMPerturbation(config.perturbation)


@pytest.fixture
def sample_features():
    """Random normalized feature vector (simulates a packet)."""
    np.random.seed(42)
    return np.random.uniform(0, 1, 256).astype(np.float32)


# ─── Correctness Tests ────────────────────────────────────────────

class TestFGSMBasic:
    def test_output_shape(self, fgsm, sample_features, model_zoo):
        """Perturbed features must have same shape as input."""
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        assert perturbed.shape == sample_features.shape
        assert delta.shape == sample_features.shape

    def test_delta_is_difference(self, fgsm, sample_features, model_zoo):
        """Delta must equal perturbed - original."""
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        np.testing.assert_allclose(perturbed - sample_features, delta, atol=1e-5)

    def test_output_dtype(self, fgsm, sample_features, model_zoo):
        """Output must be float32."""
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        assert perturbed.dtype == np.float32
        assert delta.dtype == np.float32

    def test_perturbation_is_nonzero(self, fgsm, sample_features, model_zoo):
        """A valid FGSM attack should produce a nonzero perturbation."""
        _, delta = fgsm.generate(sample_features, model_zoo)
        assert np.any(np.abs(delta) > 1e-6), "Perturbation should be nonzero"


# ─── Epsilon Bound Tests ──────────────────────────────────────────

class TestFGSMEpsilonBounds:
    def test_linf_bound_satisfied(self, fgsm, sample_features, model_zoo):
        """L∞ norm of delta must be ≤ epsilon."""
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        linf = np.max(np.abs(delta))
        assert linf <= fgsm.epsilon + 1e-5, (
            f"L∞={linf:.6f} exceeds epsilon={fgsm.epsilon}"
        )

    def test_feature_range_valid(self, fgsm, sample_features, model_zoo):
        """All perturbed features must remain in [0, 1]."""
        perturbed, _ = fgsm.generate(sample_features, model_zoo)
        assert np.all(perturbed >= -1e-5), "Features below 0"
        assert np.all(perturbed <= 1 + 1e-5), "Features above 1"

    @pytest.mark.parametrize("epsilon", [0.01, 0.03, 0.05, 0.1])
    def test_epsilon_values(self, config, sample_features, model_zoo, epsilon):
        """FGSM should respect different epsilon values."""
        config.perturbation.epsilon = epsilon
        config.perturbation.fgsm.epsilon = epsilon
        fgsm_gen = FGSMPerturbation(config.perturbation)
        perturbed, delta = fgsm_gen.generate(sample_features, model_zoo)
        linf = np.max(np.abs(delta))
        assert linf <= epsilon + 1e-5, f"epsilon={epsilon}: L∞={linf:.6f} too large"


# ─── MSE Constraint Tests ─────────────────────────────────────────

class TestFGSMMSE:
    def test_mse_below_threshold(self, fgsm, sample_features, model_zoo):
        """MSE should be at or below the threshold for small epsilon."""
        # With epsilon=0.03, MSE should be well below 0.04
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        mse = float(np.mean(delta ** 2))
        # For FGSM with epsilon=0.03: MSE = epsilon^2 = 0.0009 (much below 0.04)
        assert mse < 0.04, f"MSE={mse:.6f} exceeds 0.04 threshold"

    def test_mse_formula(self, fgsm, sample_features, model_zoo):
        """
        For FGSM (single-step sign attack), MSE ≤ epsilon².
        Since all deltas are ±epsilon: MSE = epsilon².
        """
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        mse = float(np.mean(delta ** 2))
        # Max theoretical MSE for FGSM = epsilon^2
        theoretical_max = fgsm.epsilon ** 2
        assert mse <= theoretical_max + 1e-5, (
            f"FGSM MSE={mse:.6f} exceeds theoretical max {theoretical_max:.6f}"
        )


# ─── Header Protection Tests ──────────────────────────────────────

class TestFGSMConstraints:
    def test_protected_header_dims_unchanged(self, fgsm, sample_features, model_zoo):
        """Protocol header feature dimensions [0:8] should not be modified."""
        perturbed, delta = fgsm.generate(sample_features, model_zoo)
        header_delta = np.abs(delta[:8])
        assert np.all(header_delta < 1e-5), (
            f"Header dims modified: max delta={header_delta.max():.6f}"
        )


# ─── Batch Mode Tests ─────────────────────────────────────────────

class TestFGSMBatch:
    def test_batch_output_shape(self, fgsm, model_zoo):
        """Batch output must have shape (N, feature_dim)."""
        N, D = 8, 256
        batch = np.random.uniform(0, 1, (N, D)).astype(np.float32)
        perturbed, deltas = fgsm.generate_batch(batch, model_zoo)
        assert perturbed.shape == (N, D)
        assert deltas.shape == (N, D)

    def test_batch_epsilon_bounds(self, fgsm, model_zoo):
        """All samples in batch must respect L∞ bound."""
        N = 16
        batch = np.random.uniform(0, 1, (N, 256)).astype(np.float32)
        perturbed, deltas = fgsm.generate_batch(batch, model_zoo)
        for i in range(N):
            linf = np.max(np.abs(deltas[i]))
            assert linf <= fgsm.epsilon + 1e-5, (
                f"Sample {i}: L∞={linf:.6f} > epsilon={fgsm.epsilon}"
            )

    def test_batch_consistency(self, fgsm, sample_features, model_zoo):
        """Single sample batch should give same result as individual call (approx)."""
        perturbed_single, _ = fgsm.generate(sample_features, model_zoo)
        perturbed_batch, _ = fgsm.generate_batch(sample_features[np.newaxis], model_zoo)
        # Results should be very close (may differ due to gradient computation order)
        np.testing.assert_allclose(
            perturbed_single, perturbed_batch[0], atol=1e-4
        )


# ─── Regression Tests ─────────────────────────────────────────────

class TestFGSMRegression:
    def test_zero_input(self, fgsm, model_zoo):
        """FGSM on zero vector should not crash."""
        zeros = np.zeros(256, dtype=np.float32)
        perturbed, delta = fgsm.generate(zeros, model_zoo)
        assert perturbed is not None

    def test_all_ones_input(self, fgsm, model_zoo):
        """FGSM on all-ones vector should not crash and stay in [0,1]."""
        ones = np.ones(256, dtype=np.float32)
        perturbed, delta = fgsm.generate(ones, model_zoo)
        assert np.all(perturbed <= 1 + 1e-5)

    def test_reproducible_with_same_model(self, fgsm, sample_features, model_zoo):
        """Two consecutive FGSM calls should give identical results (deterministic)."""
        p1, d1 = fgsm.generate(sample_features, model_zoo)
        p2, d2 = fgsm.generate(sample_features, model_zoo)
        np.testing.assert_array_equal(p1, p2)
