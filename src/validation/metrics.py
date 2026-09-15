"""
APE-BAIT Validation Metrics
Computes evasion rate, MSE distortion, injection latency,
throughput, and statistical significance tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ─── Detection Result ────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detection alert from a security tool."""
    tool: str
    alert_type: str
    severity: int = 1
    src_ip: str = ""
    dst_ip: str = ""
    protocol: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)


# ─── Tool Result ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Results from running a single security tool on original vs. perturbed traffic."""
    tool_name: str
    original_detections: int
    perturbed_detections: int
    evasion_rate: float          # (orig - pert) / orig
    execution_time_s: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "original_detections": self.original_detections,
            "perturbed_detections": self.perturbed_detections,
            "evasion_rate": round(self.evasion_rate, 4),
            "execution_time_s": round(self.execution_time_s, 3),
            "error": self.error,
        }


# ─── Metrics Calculator ──────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Computes all APE-BAIT validation metrics from experiment results.

    Metrics:
      - Evasion Rate per tool and aggregate
      - MSE Distortion (feature space and byte space)
      - Injection Latency (mean, p50, p95, p99)
      - Throughput (packets per second)
      - Statistical significance (paired t-test)
    """

    def __init__(self):
        self._latency_samples: List[float] = []
        self._mse_samples: List[float] = []
        self._evasion_samples: List[float] = []

    # ─── Evasion Rate ────────────────────────────────────────────

    @staticmethod
    def evasion_rate(
        original_detections: int,
        perturbed_detections: int,
    ) -> float:
        """
        Compute evasion rate for a single tool.

        Formula: (original - perturbed) / original

        Args:
            original_detections: Alerts fired on original traffic.
            perturbed_detections: Alerts fired on perturbed traffic.

        Returns:
            Evasion rate in [0.0, 1.0].
            Returns 0.0 if no original detections (no baseline to evade).
        """
        if original_detections == 0:
            return 0.0
        rate = (original_detections - perturbed_detections) / original_detections
        return float(np.clip(rate, 0.0, 1.0))

    @staticmethod
    def aggregate_evasion_rate(tool_results: List[ToolResult]) -> float:
        """Compute mean evasion rate across all tools."""
        valid = [r for r in tool_results if r.error is None and r.original_detections > 0]
        if not valid:
            return 0.0
        return float(np.mean([r.evasion_rate for r in valid]))

    # ─── Distortion ──────────────────────────────────────────────

    @staticmethod
    def mse_distortion(original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Compute Mean Squared Error between original and perturbed features.

        Args:
            original: Original feature vector or byte array (normalized [0,1]).
            perturbed: Perturbed feature vector or byte array.

        Returns:
            MSE value. Target: < 0.04 for human-invisible threshold.
        """
        return float(np.mean((original.astype(np.float64) - perturbed.astype(np.float64)) ** 2))

    @staticmethod
    def mse_bytes(original_bytes: bytes, perturbed_bytes: bytes) -> float:
        """Compute MSE directly on raw packet bytes."""
        max_len = max(len(original_bytes), len(perturbed_bytes))
        orig_arr = np.frombuffer(original_bytes.ljust(max_len, b'\x00'), dtype=np.uint8) / 255.0
        pert_arr = np.frombuffer(perturbed_bytes.ljust(max_len, b'\x00'), dtype=np.uint8) / 255.0
        return float(np.mean((orig_arr - pert_arr) ** 2))

    @staticmethod
    def linf_distortion(original: np.ndarray, perturbed: np.ndarray) -> float:
        """Compute L∞ norm of perturbation."""
        return float(np.max(np.abs(original - perturbed)))

    @staticmethod
    def l2_distortion(original: np.ndarray, perturbed: np.ndarray) -> float:
        """Compute L2 norm of perturbation."""
        return float(np.linalg.norm(original - perturbed))

    # ─── Latency ─────────────────────────────────────────────────

    @staticmethod
    def latency_stats(latency_samples_ms: List[float]) -> Dict[str, float]:
        """
        Compute latency statistics from a list of per-packet latency measurements.

        Returns dict with: mean, median, p95, p99, min, max (all in ms).
        """
        if not latency_samples_ms:
            return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        arr = np.array(latency_samples_ms)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # ─── Throughput ───────────────────────────────────────────────

    @staticmethod
    def throughput_pps(num_packets: int, elapsed_seconds: float) -> float:
        """Compute packets-per-second throughput."""
        if elapsed_seconds <= 0:
            return 0.0
        return float(num_packets / elapsed_seconds)

    # ─── Statistical Tests ────────────────────────────────────────

    @staticmethod
    def paired_ttest(
        original_detections: List[int],
        perturbed_detections: List[int],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Paired t-test to determine if evasion is statistically significant.

        Args:
            original_detections: Detection counts on original traffic samples.
            perturbed_detections: Detection counts on perturbed traffic samples.
            alpha: Significance level.

        Returns:
            Dict with t_statistic, p_value, significant, effect_size (Cohen's d).
        """
        orig = np.array(original_detections, dtype=np.float64)
        pert = np.array(perturbed_detections, dtype=np.float64)

        if len(orig) < 2:
            return {"t_statistic": 0, "p_value": 1.0, "significant": False, "effect_size": 0}

        t_stat, p_val = stats.ttest_rel(orig, pert)

        # Cohen's d for paired data
        diff = orig - pert
        effect_size = float(np.mean(diff) / (np.std(diff) + 1e-8))

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < alpha,
            "effect_size_cohens_d": effect_size,
            "alpha": alpha,
        }

    # ─── Summary Report ───────────────────────────────────────────

    @staticmethod
    def full_report(
        tool_results: List[ToolResult],
        latency_samples_ms: List[float],
        mse_samples: List[float],
    ) -> Dict[str, Any]:
        """
        Generate a complete metrics summary report.

        Args:
            tool_results: Results from each validated security tool.
            latency_samples_ms: Per-packet latency measurements.
            mse_samples: Per-packet MSE distortion measurements.

        Returns:
            Nested dict with all metrics for the report.
        """
        mc = MetricsCalculator()

        evasion_by_tool = {r.tool_name: r.to_dict() for r in tool_results}
        agg_evasion = mc.aggregate_evasion_rate(tool_results)

        lat_stats = mc.latency_stats(latency_samples_ms)

        mse_arr = np.array(mse_samples) if mse_samples else np.array([0.0])

        return {
            "summary": {
                "aggregate_evasion_rate": round(agg_evasion, 4),
                "tools_tested": len(tool_results),
                "tools_evaded": sum(1 for r in tool_results if r.evasion_rate >= 0.5),
                "avg_mse": round(float(np.mean(mse_arr)), 6),
                "max_mse": round(float(np.max(mse_arr)), 6),
                "mse_below_threshold": bool(np.mean(mse_arr) < 0.04),
                "avg_latency_ms": round(lat_stats.get("mean", 0), 2),
                "p95_latency_ms": round(lat_stats.get("p95", 0), 2),
                "latency_below_2s": bool(lat_stats.get("p95", 999) < 2000),
            },
            "evasion_by_tool": evasion_by_tool,
            "latency": lat_stats,
            "distortion": {
                "mean_mse": round(float(np.mean(mse_arr)), 6),
                "p95_mse": round(float(np.percentile(mse_arr, 95)), 6),
                "max_mse": round(float(np.max(mse_arr)), 6),
                "samples": len(mse_samples),
            },
        }
