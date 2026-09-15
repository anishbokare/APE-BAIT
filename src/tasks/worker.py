"""
APE-BAIT Celery Worker Tasks.

Runs the perturbation + validation pipeline as a background task.
Progress updates are stored via Celery's result backend (Redis).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from celery import Task
from celery.utils.log import get_task_logger

from src.tasks.celery_app import celery_app
from src.core.config import load_config
from src.perturbation.fgsm import FGSMPerturbation
from src.perturbation.pgd import PGDPerturbation
from src.perturbation.combined import EnsemblePerturbation
from src.injection.injector import TrafficInjector
from src.validation.runner import ValidationRunner
from src.validation.metrics import MetricsCalculator
from src.capture.preprocessor import PacketPreprocessor

logger = get_task_logger(__name__)

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PerturbationTask(Task):
    """Base task that holds long-lived engine objects across calls."""
    abstract = True
    _cfg = None
    _preprocessor = None

    @property
    def cfg(self):
        if self._cfg is None:
            self._cfg = load_config()
        return self._cfg

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = PacketPreprocessor(self.cfg)
        return self._preprocessor


@celery_app.task(bind=True, base=PerturbationTask, name="ape_bait.run_perturbation_job")
def run_perturbation_job(self: PerturbationTask, job_id: str, pcap_path: str, params_dict: dict) -> dict:
    """
    Main Celery task: perturb traffic and validate against target tools.

    Args:
        job_id:     UUID string for this job.
        pcap_path:  Path to the input PCAP, or "live://<iface>:<duration>".
        params_dict: Serialised AttackParams dict.

    Returns:
        dict with summary, tool_results, perturbed_pcap_url, report_url.
    """
    self.update_state(state="PROGRESS", meta={"progress": 5, "message": "Loading configuration"})
    cfg = self.cfg
    method = params_dict["method"]
    epsilon = params_dict["epsilon"]
    alpha = params_dict.get("alpha", 0.005)
    iterations = params_dict.get("iterations", 10)
    target_tools = params_dict.get("target_tools", [])

    logger.info(f"[{job_id}] Starting {method.upper()} job: epsilon={epsilon}, iters={iterations}")

    # ── Step 1: Load and preprocess traffic ──────────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Loading traffic"})

    if pcap_path.startswith("live://"):
        # Live capture (stub — requires root / Scapy)
        _, rest = pcap_path.split("://", 1)
        iface, duration = rest.split(":")
        logger.info(f"[{job_id}] Live capture on {iface} for {duration}s")
        from scapy.all import sniff
        pkts = sniff(iface=iface, timeout=int(duration))
    else:
        from scapy.all import rdpcap
        pkts = rdpcap(pcap_path)

    logger.info(f"[{job_id}] Loaded {len(pkts)} packets")
    self.update_state(state="PROGRESS", meta={"progress": 20, "message": f"Loaded {len(pkts)} packets"})

    # ── Step 2: Feature extraction ────────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Extracting features"})
    feature_vectors = self.preprocessor.packets_to_features(pkts)

    # ── Step 3: Adversarial perturbation ─────────────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 40, "message": f"Running {method.upper()} perturbation"})

    if method == "fgsm":
        engine = FGSMPerturbation(cfg)
        perturbed_vectors = engine.perturb_batch(feature_vectors, epsilon=epsilon)
    elif method == "pgd":
        engine = PGDPerturbation(cfg)
        perturbed_vectors = engine.perturb_batch(feature_vectors, epsilon=epsilon, alpha=alpha, steps=iterations)
    else:
        engine = EnsemblePerturbation(cfg)
        perturbed_vectors = engine.perturb_batch(feature_vectors, epsilon=epsilon, alpha=alpha, steps=iterations)

    logger.info(f"[{job_id}] Perturbation complete for {len(perturbed_vectors)} feature vectors")

    # ── Step 4: Inject perturbations into packets ─────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 55, "message": "Injecting perturbations into packets"})

    injector = TrafficInjector(cfg)
    injection_results = []
    perturbed_pkts = []

    for i, (pkt, orig_vec, pert_vec) in enumerate(zip(pkts, feature_vectors, perturbed_vectors)):
        t0 = time.perf_counter()
        perturbed_pkt = injector.inject(pkt, orig_vec, pert_vec)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        perturbed_pkts.append(perturbed_pkt)
        injection_results.append({"latency_ms": elapsed_ms, "orig_vec": orig_vec, "pert_vec": pert_vec})

    # Write perturbed PCAP
    perturbed_pcap_path = RESULTS_DIR / f"{job_id}_perturbed.pcap"
    injector.write_pcap(perturbed_pkts, str(perturbed_pcap_path))
    logger.info(f"[{job_id}] Wrote perturbed PCAP: {perturbed_pcap_path}")

    # ── Step 5: Metrics computation ───────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 65, "message": "Computing distortion metrics"})

    mc = MetricsCalculator(cfg)
    latency_samples = [r["latency_ms"] for r in injection_results]
    mse_samples = [
        mc.mse(r["orig_vec"], r["pert_vec"])
        for r in injection_results
    ]
    latency_stats = mc.latency_stats(latency_samples)
    distortion_stats = mc.distortion_stats(mse_samples)

    # ── Step 6: Validation against target tools ───────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 75, "message": f"Running validation against {len(target_tools)} tools"})

    runner = ValidationRunner(cfg)
    validation_report = runner.run(
        original_pcap=pcap_path if not pcap_path.startswith("live://") else str(perturbed_pcap_path),
        perturbed_pcap=str(perturbed_pcap_path),
        tools=target_tools if target_tools else None,
    )

    # ── Step 7: Aggregate and save report ────────────────────────────────────
    self.update_state(state="PROGRESS", meta={"progress": 90, "message": "Generating report"})

    evasion_by_tool = validation_report.get("evasion_by_tool", {})
    tool_results = []
    for tool_name, result in evasion_by_tool.items():
        tool_results.append({
            "tool": tool_name,
            "original_detections": result.get("original_detections", 0),
            "perturbed_detections": result.get("perturbed_detections", 0),
            "evasion_rate": result.get("evasion_rate", 0.0),
            "execution_time_s": result.get("execution_time_s", 0.0),
            "error": result.get("error"),
        })

    aggregate_rate = sum(r["evasion_rate"] for r in tool_results) / max(len(tool_results), 1)
    tools_evaded = sum(1 for r in tool_results if r["evasion_rate"] >= 0.5)

    summary = {
        "aggregate_evasion_rate": round(aggregate_rate, 4),
        "tools_tested": len(tool_results),
        "tools_evaded": tools_evaded,
        "distortion": {
            "mean_mse": round(distortion_stats["mean"], 6),
            "p95_mse": round(distortion_stats["p95"], 6),
            "max_mse": round(distortion_stats["max"], 6),
            "below_threshold": distortion_stats["mean"] < 0.04,
        },
        "latency": {
            "mean_ms": round(latency_stats["mean"], 2),
            "p50_ms": round(latency_stats["p50"], 2),
            "p95_ms": round(latency_stats["p95"], 2),
            "p99_ms": round(latency_stats.get("p99", latency_stats["p95"]), 2),
            "below_budget": latency_stats["p95"] < 2000,
        },
    }

    full_report = {
        "job_id": job_id,
        "summary": summary,
        "tool_results": tool_results,
        "evasion_by_tool": evasion_by_tool,
        "attack_params": params_dict,
    }

    report_path = RESULTS_DIR / f"{job_id}_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"[{job_id}] Complete. Evasion: {aggregate_rate:.1%} | MSE: {distortion_stats['mean']:.4f}")

    return {
        "summary": summary,
        "tool_results": tool_results,
        "perturbed_pcap_url": f"/api/v1/jobs/{job_id}/pcap",
        "report_url": f"/api/v1/jobs/{job_id}/report",
        "progress": 100,
        "progress_message": "Complete",
    }
