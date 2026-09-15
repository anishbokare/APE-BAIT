"""
Unit tests for validation metrics calculations.
Tests evasion rate, MSE, latency stats, and statistical significance.
"""

import numpy as np
import pytest

from src.validation.metrics import MetricsCalculator, ToolResult


@pytest.fixture
def mc():
    return MetricsCalculator()


# ─── Evasion Rate Tests ───────────────────────────────────────────

class TestEvasionRate:
    def test_perfect_evasion(self, mc):
        """If no detections after perturbation, evasion = 1.0."""
        assert mc.evasion_rate(10, 0) == 1.0

    def test_zero_evasion(self, mc):
        """If same detections, evasion = 0.0."""
        assert mc.evasion_rate(10, 10) == 0.0

    def test_partial_evasion(self, mc):
        """Partial evasion should compute correctly."""
        rate = mc.evasion_rate(20, 4)
        assert abs(rate - 0.8) < 1e-6

    def test_no_original_detections(self, mc):
        """If no original detections, return 0 (no baseline)."""
        assert mc.evasion_rate(0, 0) == 0.0
        assert mc.evasion_rate(0, 5) == 0.0  # Shouldn't happen, but handled

    def test_evasion_clamped(self, mc):
        """Evasion rate must not exceed 1.0 or go below 0.0."""
        r1 = mc.evasion_rate(5, 0)   # 1.0
        r2 = mc.evasion_rate(5, 10)  # Would be negative, clamped to 0
        assert 0 <= r1 <= 1
        assert 0 <= r2 <= 1

    def test_aggregate_evasion(self, mc):
        """Aggregate evasion is mean across tools with valid baselines."""
        results = [
            ToolResult("snort", 10, 2, mc.evasion_rate(10, 2), 1.0),
            ToolResult("suricata", 20, 4, mc.evasion_rate(20, 4), 1.5),
            ToolResult("yara", 0, 0, 0.0, 0.5),   # No baseline, excluded
            ToolResult("clamav", 15, 3, mc.evasion_rate(15, 3), 2.0, error=None),
        ]
        agg = mc.aggregate_evasion_rate(results)
        # Expected: mean of [0.8, 0.8, 0.8] = 0.8
        assert abs(agg - 0.8) < 1e-6

    def test_aggregate_all_zero_baseline(self, mc):
        """Aggregate evasion with all-zero baselines should return 0."""
        results = [
            ToolResult("tool1", 0, 0, 0.0, 1.0),
            ToolResult("tool2", 0, 0, 0.0, 1.0),
        ]
        agg = mc.aggregate_evasion_rate(results)
        assert agg == 0.0

    def test_aggregate_excludes_errors(self, mc):
        """Tools with errors should be excluded from aggregate."""
        results = [
            ToolResult("snort", 10, 2, 0.8, 1.0, error=None),
            ToolResult("crash", 0, 0, 0.0, 0.0, error="Tool crashed"),
        ]
        agg = mc.aggregate_evasion_rate(results)
        assert abs(agg - 0.8) < 1e-6


# ─── MSE Tests ───────────────────────────────────────────────────

class TestMSEDistortion:
    def test_identical_vectors(self, mc):
        """MSE between identical vectors is 0."""
        v = np.random.rand(256).astype(np.float32)
        assert mc.mse_distortion(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_known_mse(self, mc):
        """Verify MSE calculation with known values."""
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        mse = mc.mse_distortion(a, b)
        assert abs(mse - 0.5) < 1e-6

    def test_below_threshold(self, mc):
        """Small epsilon perturbation should give MSE << 0.04."""
        orig = np.random.rand(256).astype(np.float32)
        eps = 0.03
        perturbed = orig + np.random.uniform(-eps, eps, 256).astype(np.float32)
        perturbed = np.clip(perturbed, 0, 1)
        mse = mc.mse_distortion(orig, perturbed)
        # With uniform noise in [-0.03, 0.03]: E[δ²] = (0.03)²/3 ≈ 0.0003
        assert mse < 0.04, f"Small perturbation MSE={mse:.6f} exceeds threshold"

    def test_mse_bytes(self, mc):
        """MSE on byte arrays should be similar to normalized feature MSE."""
        a = b'\x00' * 100
        b_arr = b'\x0a' * 100
        mse = mc.mse_bytes(a, b_arr)
        expected = (10 / 255.0) ** 2
        assert abs(mse - expected) < 1e-4


# ─── Latency Tests ───────────────────────────────────────────────

class TestLatencyStats:
    def test_empty_samples(self, mc):
        """Empty latency list should return zeros."""
        stats = mc.latency_stats([])
        assert stats["mean"] == 0
        assert stats["p95"] == 0

    def test_single_sample(self, mc):
        """Single sample: all stats should equal that sample."""
        stats = mc.latency_stats([500.0])
        assert stats["mean"] == pytest.approx(500.0)
        assert stats["min"] == pytest.approx(500.0)
        assert stats["max"] == pytest.approx(500.0)

    def test_percentiles(self, mc):
        """P95 should be close to the 95th percentile."""
        samples = list(np.arange(1, 101, dtype=float))  # 1..100 ms
        stats = mc.latency_stats(samples)
        assert abs(stats["p95"] - 95.5) < 1.0  # scipy percentile

    def test_latency_below_2s(self, mc):
        """Check latency budget compliance."""
        samples = np.random.uniform(100, 1500, 100).tolist()
        stats = mc.latency_stats(samples)
        assert stats["p95"] < 2000, "p95 should be below 2000ms for this test data"

    def test_std_nonzero(self, mc):
        """Std dev should be nonzero for varied samples."""
        samples = [100, 200, 300, 400, 500]
        stats = mc.latency_stats(samples)
        assert stats["std"] > 0


# ─── Throughput Tests ─────────────────────────────────────────────

class TestThroughput:
    def test_basic_throughput(self, mc):
        assert mc.throughput_pps(1000, 1.0) == pytest.approx(1000.0)

    def test_zero_elapsed(self, mc):
        assert mc.throughput_pps(1000, 0.0) == 0.0

    def test_fractional(self, mc):
        assert mc.throughput_pps(500, 0.5) == pytest.approx(1000.0)


# ─── Statistical Tests ────────────────────────────────────────────

class TestStatisticalTests:
    def test_significant_evasion(self, mc):
        """Significant reduction in detections should yield p < 0.05."""
        orig = [10, 12, 11, 9, 13, 10, 11, 12, 10, 11]
        pert = [1, 2, 1, 0, 2, 1, 1, 2, 0, 1]
        result = mc.paired_ttest(orig, pert)
        assert result["p_value"] < 0.05
        assert result["significant"] == True

    def test_no_evasion_not_significant(self, mc):
        """No change in detections should not be significant."""
        same = [5, 6, 5, 7, 5, 6, 5, 5]
        result = mc.paired_ttest(same, same)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01) or not result["significant"]

    def test_single_sample_handled(self, mc):
        """Single sample should not crash."""
        result = mc.paired_ttest([10], [2])
        assert "p_value" in result

    def test_effect_size_large_when_significant(self, mc):
        """Cohen's d should be large when evasion is strong."""
        orig = [20] * 20
        pert = [1] * 20
        result = mc.paired_ttest(orig, pert)
        assert abs(result["effect_size_cohens_d"]) > 1.0  # Large effect


# ─── Full Report Test ─────────────────────────────────────────────

class TestFullReport:
    def test_report_structure(self, mc):
        """Full report should have expected top-level keys."""
        tool_results = [
            ToolResult("snort", 10, 2, 0.8, 1.0),
            ToolResult("suricata", 8, 1, 0.875, 1.2),
        ]
        latency = np.random.uniform(500, 1000, 50).tolist()
        mse = np.random.uniform(0.01, 0.03, 50).tolist()
        report = mc.full_report(tool_results, latency, mse)

        assert "summary" in report
        assert "evasion_by_tool" in report
        assert "latency" in report
        assert "distortion" in report

    def test_report_summary_values(self, mc):
        """Summary should correctly aggregate results."""
        tool_results = [
            ToolResult("a", 10, 2, 0.8, 1.0),
            ToolResult("b", 10, 2, 0.8, 1.0),
        ]
        report = mc.full_report(tool_results, [500, 600], [0.02, 0.03])
        summary = report["summary"]
        assert abs(summary["aggregate_evasion_rate"] - 0.8) < 1e-6
        assert summary["tools_tested"] == 2
        assert summary["mse_below_threshold"] is True
