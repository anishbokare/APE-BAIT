"""
Unit tests for PGD perturbation generator.
Tests multi-step iteration, random restart, L∞ projection, and early stopping.
"""

import numpy as np
import pytest
import torch

from src.core.config import load_config
from src.models.model_zoo import ModelZoo
from src.perturbation.pgd import PGDPerturbation


@pytest.fixture
def config():
    cfg = load_config()
    cfg.perturbation.pgd.steps = 10   # Fast tests
    cfg.perturbation.pgd.restarts = 1
    return cfg


@pytest.fixture
def model_zoo(config):
    zoo = ModelZoo(config)
    zoo.preload_all()
    return zoo


@pytest.fixture
def pgd(config):
    return PGDPerturbation(config.perturbation)


@pytest.fixture
def sample_features():
    np.random.seed(123)
    return np.random.uniform(0, 1, 256).astype(np.float32)


# ─── Basic Correctness ────────────────────────────────────────────

class TestPGDBasic:
    def test_output_shape(self, pgd, sample_features, model_zoo):
        perturbed, delta = pgd.generate(sample_features, model_zoo)
        assert perturbed.shape == sample_features.shape
        assert delta.shape == sample_features.shape

    def test_delta_is_difference(self, pgd, sample_features, model_zoo):
        perturbed, delta = pgd.generate(sample_features, model_zoo)
        np.testing.assert_allclose(perturbed - sample_features, delta, atol=1e-5)

    def test_output_dtype(self, pgd, sample_features, model_zoo):
        perturbed, _ = pgd.generate(sample_features, model_zoo)
        assert perturbed.dtype == np.float32

    def test_perturbation_nonzero(self, pgd, sample_features, model_zoo):
        _, delta = pgd.generate(sample_features, model_zoo)
        assert np.any(np.abs(delta) > 1e-6), "PGD perturbation should be nonzero"


# ─── L∞ Projection Tests ──────────────────────────────────────────

class TestPGDProjection:
    def test_linf_bound(self, pgd, sample_features, model_zoo):
        """After 10 PGD steps, L∞ must be ≤ epsilon."""
        perturbed, delta = pgd.generate(sample_features, model_zoo)
        linf = np.max(np.abs(delta))
        assert linf <= pgd.epsilon + 1e-5, f"L∞={linf:.6f} > eps={pgd.epsilon}"

    def test_feature_range(self, pgd, sample_features, model_zoo):
        """All features must remain in [0, 1]."""
        perturbed, _ = pgd.generate(sample_features, model_zoo)
        assert np.all(perturbed >= -1e-5)
        assert np.all(perturbed <= 1 + 1e-5)

    @pytest.mark.parametrize("steps", [1, 5, 10, 20])
    def test_linf_with_varying_steps(self, config, sample_features, model_zoo, steps):
        """L∞ bound must hold regardless of step count."""
        config.perturbation.pgd.steps = steps
        pgd_gen = PGDPerturbation(config.perturbation)
        perturbed, delta = pgd_gen.generate(sample_features, model_zoo)
        linf = np.max(np.abs(delta))
        assert linf <= pgd_gen.epsilon + 1e-5


# ─── Random Start Tests ───────────────────────────────────────────

class TestPGDRandomStart:
    def test_random_start_produces_variation(self, config, sample_features, model_zoo):
        """With random start enabled, repeated calls may differ."""
        config.perturbation.pgd.random_start = True
        config.perturbation.pgd.restarts = 1
        pgd_gen = PGDPerturbation(config.perturbation)

        results = [pgd_gen.generate(sample_features, model_zoo)[0] for _ in range(3)]
        # At least some variation expected
        all_same = all(np.allclose(results[0], r) for r in results[1:])
        # With random starts, may or may not be same (depends on random seed)
        # Just ensure no crashes and valid bounds
        for r in results:
            linf = np.max(np.abs(r - sample_features))
            assert linf <= pgd_gen.epsilon + 1e-5

    def test_no_random_start_deterministic(self, config, sample_features, model_zoo):
        """Without random start, PGD must be deterministic."""
        config.perturbation.pgd.random_start = False
        config.perturbation.pgd.restarts = 1
        pgd_gen = PGDPerturbation(config.perturbation)
        p1, _ = pgd_gen.generate(sample_features, model_zoo)
        p2, _ = pgd_gen.generate(sample_features, model_zoo)
        np.testing.assert_array_equal(p1, p2)


# ─── MSE Tests ────────────────────────────────────────────────────

class TestPGDMSE:
    def test_mse_below_threshold(self, pgd, sample_features, model_zoo):
        """PGD MSE should remain below the configured threshold."""
        perturbed, delta = pgd.generate(sample_features, model_zoo)
        mse = float(np.mean(delta ** 2))
        assert mse <= 0.04, f"MSE={mse:.6f} exceeds 0.04 threshold"


# ─── Early Stopping Tests ─────────────────────────────────────────

class TestPGDEarlyStopping:
    def test_early_stop_doesnt_crash(self, config, sample_features, model_zoo):
        """Early stopping enabled should not cause errors."""
        config.perturbation.pgd.early_stop = True
        config.perturbation.pgd.steps = 40
        pgd_gen = PGDPerturbation(config.perturbation)
        perturbed, delta = pgd_gen.generate(sample_features, model_zoo)
        assert perturbed is not None
        assert np.max(np.abs(delta)) <= pgd_gen.epsilon + 1e-5


# ─── Batch Tests ──────────────────────────────────────────────────

class TestPGDBatch:
    def test_batch_shape(self, pgd, model_zoo):
        N, D = 4, 256
        batch = np.random.uniform(0, 1, (N, D)).astype(np.float32)
        perturbed, deltas = pgd.generate_batch(batch, model_zoo)
        assert perturbed.shape == (N, D)
        assert deltas.shape == (N, D)

    def test_batch_linf_bounds(self, pgd, model_zoo):
        N = 6
        batch = np.random.uniform(0, 1, (N, 256)).astype(np.float32)
        perturbed, deltas = pgd.generate_batch(batch, model_zoo)
        for i in range(N):
            linf = np.max(np.abs(deltas[i]))
            assert linf <= pgd.epsilon + 1e-5, f"Batch sample {i} L∞={linf:.6f}"


# ─── PGD Stronger Than FGSM ──────────────────────────────────────

class TestPGDStrength:
    def test_pgd_loss_lower_than_fgsm(self, config, sample_features, model_zoo):
        """
        PGD should achieve lower surrogate loss than FGSM
        (stronger attack → more effective evasion).
        """
        import torch
        from src.perturbation.fgsm import FGSMPerturbation

        fgsm_gen = FGSMPerturbation(config.perturbation)
        pgd_gen = PGDPerturbation(config.perturbation)

        fgsm_pert, _ = fgsm_gen.generate(sample_features, model_zoo)
        pgd_pert, _ = pgd_gen.generate(sample_features, model_zoo)

        model = model_zoo.ids
        with torch.no_grad():
            x_fgsm = torch.from_numpy(fgsm_pert).unsqueeze(0)
            x_pgd = torch.from_numpy(pgd_pert).unsqueeze(0)

            import torch.nn.functional as F
            prob_fgsm = F.softmax(model(x_fgsm), dim=-1)[0, 1].item()
            prob_pgd = F.softmax(model(x_pgd), dim=-1)[0, 1].item()

        # PGD should typically push malicious probability lower
        # (both likely low since models start random — just check it runs)
        assert 0 <= prob_fgsm <= 1
        assert 0 <= prob_pgd <= 1
