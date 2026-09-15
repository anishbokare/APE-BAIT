"""
APE-BAIT Validation Runner
Orchestrates validation testing of perturbed traffic against 15+ security tools.
Runs original vs. perturbed PCAP through each tool, computes evasion metrics.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import APEBAITConfig, ValidationConfig
from src.core.logger import get_logger
from src.validation.metrics import Detection, MetricsCalculator, ToolResult
from src.validation.tool_adapters import ADAPTER_REGISTRY, BaseToolAdapter, get_adapter


class ValidationRunner:
    """
    Orchestrates validation experiments against configured security tools.

    Workflow:
      1. Load original PCAP and perturbed PCAP
      2. Run each configured tool on both PCAPs
      3. Compare detections: compute per-tool evasion rate
      4. Aggregate metrics and return full report
    """

    def __init__(self, config: APEBAITConfig):
        self.config = config
        self.val_config = config.validation
        self.logger = get_logger("ape-bait.validation")
        self.metrics_calc = MetricsCalculator()

    def run(
        self,
        original_pcap: str,
        perturbed_pcap: str,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run full validation suite.

        Args:
            original_pcap: Path to original (unperturbed) traffic PCAP.
            perturbed_pcap: Path to adversarially perturbed traffic PCAP.
            tools: Optional list of tool names to test. Defaults to config.

        Returns:
            Full validation report dict.
        """
        tools = tools or self.val_config.tools
        self.logger.info(
            "Starting validation run",
            tools=tools,
            original_pcap=original_pcap,
            perturbed_pcap=perturbed_pcap,
        )

        # Validate paths
        for path in [original_pcap, perturbed_pcap]:
            if not Path(path).exists():
                raise FileNotFoundError(f"PCAP not found: {path}")

        # Run tools
        if self.val_config.parallel:
            tool_results = self._run_parallel(tools, original_pcap, perturbed_pcap)
        else:
            tool_results = self._run_sequential(tools, original_pcap, perturbed_pcap)

        # Compile report
        report = self.metrics_calc.full_report(
            tool_results=tool_results,
            latency_samples_ms=[],   # Latency tracked by engine
            mse_samples=[],          # MSE tracked by engine
        )

        # Log summary
        summary = report["summary"]
        self.logger.info(
            "Validation complete",
            tools_tested=summary["tools_tested"],
            agg_evasion_rate=summary["aggregate_evasion_rate"],
            tools_evaded=summary["tools_evaded"],
        )

        return report

    def run_single_tool(
        self,
        tool_name: str,
        original_pcap: str,
        perturbed_pcap: str,
    ) -> ToolResult:
        """
        Run validation for a single tool.

        Args:
            tool_name: Name of the tool (from ADAPTER_REGISTRY).
            original_pcap: Path to original PCAP.
            perturbed_pcap: Path to perturbed PCAP.

        Returns:
            ToolResult with evasion metrics.
        """
        start = time.time()
        adapter = self._get_adapter(tool_name)

        if not adapter.is_available():
            self.logger.warning(f"Tool not available: {tool_name}")
            return ToolResult(
                tool_name=tool_name,
                original_detections=0,
                perturbed_detections=0,
                evasion_rate=0.0,
                execution_time_s=0.0,
                error="Tool not available",
            )

        # Run on original traffic
        try:
            orig_detections = adapter.analyze(original_pcap)
            self.logger.info(
                f"[{tool_name}] Original: {len(orig_detections)} detections"
            )
        except Exception as e:
            self.logger.error(f"[{tool_name}] Error on original PCAP: {e}")
            return ToolResult(
                tool_name=tool_name,
                original_detections=0,
                perturbed_detections=0,
                evasion_rate=0.0,
                execution_time_s=time.time() - start,
                error=str(e),
            )

        # Run on perturbed traffic
        try:
            pert_detections = adapter.analyze(perturbed_pcap)
            self.logger.info(
                f"[{tool_name}] Perturbed: {len(pert_detections)} detections"
            )
        except Exception as e:
            self.logger.error(f"[{tool_name}] Error on perturbed PCAP: {e}")
            return ToolResult(
                tool_name=tool_name,
                original_detections=len(orig_detections),
                perturbed_detections=0,
                evasion_rate=0.0,
                execution_time_s=time.time() - start,
                error=str(e),
            )

        evasion = self.metrics_calc.evasion_rate(
            len(orig_detections), len(pert_detections)
        )

        return ToolResult(
            tool_name=tool_name,
            original_detections=len(orig_detections),
            perturbed_detections=len(pert_detections),
            evasion_rate=evasion,
            execution_time_s=time.time() - start,
        )

    # ─── Internal ───────────────────────────────────────────────

    def _run_sequential(
        self,
        tools: List[str],
        original_pcap: str,
        perturbed_pcap: str,
    ) -> List[ToolResult]:
        results = []
        for tool in tools:
            self.logger.info(f"Running tool: {tool}")
            result = self.run_single_tool(tool, original_pcap, perturbed_pcap)
            results.append(result)
        return results

    def _run_parallel(
        self,
        tools: List[str],
        original_pcap: str,
        perturbed_pcap: str,
        max_workers: int = 8,
    ) -> List[ToolResult]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_single_tool, tool, original_pcap, perturbed_pcap): tool
                for tool in tools
            }
            for future in concurrent.futures.as_completed(futures):
                tool_name = futures[future]
                try:
                    result = future.result(timeout=self.val_config.timeout_per_tool)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Tool {tool_name} failed: {e}")
                    results.append(ToolResult(
                        tool_name=tool_name,
                        original_detections=0,
                        perturbed_detections=0,
                        evasion_rate=0.0,
                        execution_time_s=0.0,
                        error=str(e),
                    ))
        return results

    def _get_adapter(self, tool_name: str) -> BaseToolAdapter:
        """Instantiate the adapter for a given tool name."""
        kwargs: Dict[str, Any] = {
            "feature_dim": self.config.models.feature_dim,
        }
        return get_adapter(tool_name, **{k: v for k, v in kwargs.items()
                           if _adapter_accepts(tool_name, k)})


def _adapter_accepts(tool_name: str, kwarg: str) -> bool:
    """Check if an adapter class's __init__ accepts a given kwarg."""
    import inspect
    cls = ADAPTER_REGISTRY.get(tool_name)
    if cls is None:
        return False
    sig = inspect.signature(cls.__init__)
    return kwarg in sig.parameters
