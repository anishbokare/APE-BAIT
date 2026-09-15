"""
APE-BAIT Main Engine Orchestrator
Coordinates the full pipeline: capture → preprocess → perturb → validate → inject.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np

from src.core.config import APEBAITConfig, load_config
from src.core.logger import get_logger, timing


# ─── Runtime Metrics ────────────────────────────────────────────────────────

@dataclass
class EngineMetrics:
    """Live runtime metrics collected by the engine."""
    packets_captured: int = 0
    packets_perturbed: int = 0
    packets_injected: int = 0
    evasion_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_mse: float = 0.0
    throughput_pps: float = 0.0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    latency_history: List[float] = field(default_factory=list)
    mse_history: List[float] = field(default_factory=list)

    def update_latency(self, ms: float) -> None:
        self.latency_history.append(ms)
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        self.avg_latency_ms = float(np.mean(self.latency_history))

    def update_mse(self, mse: float) -> None:
        self.mse_history.append(mse)
        if len(self.mse_history) > 1000:
            self.mse_history = self.mse_history[-1000:]
        self.avg_mse = float(np.mean(self.mse_history))

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packets_captured": self.packets_captured,
            "packets_perturbed": self.packets_perturbed,
            "packets_injected": self.packets_injected,
            "evasion_rate": round(self.evasion_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_mse": round(self.avg_mse, 6),
            "throughput_pps": round(self.throughput_pps, 1),
            "errors": self.errors,
            "uptime_seconds": round(self.uptime_seconds(), 1),
        }


# ─── APE-BAIT Engine ────────────────────────────────────────────────────────

class APEBAITEngine:
    """
    Main orchestrator for the Adversarial Perturbation Engine.

    Pipeline:
      1. PacketSniffer captures raw packets → queue
      2. PacketPreprocessor converts to feature tensors
      3. PerturbationEngine (FGSM/PGD) generates adversarial delta
      4. StealthValidator checks MSE < threshold
      5. TrafficInjector re-emits perturbed packets
    """

    def __init__(self, config: APEBAITConfig):
        self.config = config
        self.logger = get_logger(
            "ape-bait.engine",
            level=config.engine.log_level,
        )
        self.metrics = EngineMetrics()
        self._running = threading.Event()
        self._packet_queue: queue.Queue = queue.Queue(
            maxsize=config.engine.queue_size
        )

        # Lazy-initialize heavy components
        self._sniffer = None
        self._preprocessor = None
        self._perturber = None
        self._injector = None
        self._models = None

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor
        from src.models.model_zoo import ModelZoo
        from src.perturbation.combined import EnsemblePerturbation
        from src.perturbation.fgsm import FGSMPerturbation
        from src.perturbation.pgd import PGDPerturbation
        from src.injection.injector import TrafficInjector

        self.logger.info("Initializing APE-BAIT pipeline components...")

        # Capture
        self._sniffer = PacketSniffer(
            interface=self.config.engine.interface,
            bpf_filter=self.config.capture.bpf_filter,
            snapshot_len=self.config.capture.snapshot_len,
        )
        self._preprocessor = PacketPreprocessor(
            feature_dim=self.config.models.feature_dim,
            payload_max_len=self.config.models.payload_max_len,
        )

        # Surrogate models
        self._models = ModelZoo(self.config)

        # Perturbation engine
        method = self.config.perturbation.method
        if method == "fgsm":
            self._perturber = FGSMPerturbation(self.config.perturbation)
        elif method == "pgd":
            self._perturber = PGDPerturbation(self.config.perturbation)
        elif method == "ensemble":
            self._perturber = EnsemblePerturbation(self.config.perturbation)
        else:
            raise ValueError(f"Unknown perturbation method: {method}")

        # Injection
        self._injector = TrafficInjector(self.config.injection)

        self.logger.info(
            "Pipeline initialized",
            method=method,
            device=self.config.models.device,
        )

    def run_demo(self, pcap_path: str) -> Dict[str, Any]:
        """
        Run the engine in demo mode: replay a PCAP file, perturb traffic,
        write perturbed PCAP, return metrics.
        """
        from src.capture.sniffer import PacketSniffer

        self.logger.info("Starting demo mode", pcap=pcap_path)
        self._init_components()

        pcap_file = Path(pcap_path)
        if not pcap_file.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        packets = self._sniffer.read_pcap(pcap_path)
        self.logger.info(f"Loaded {len(packets)} packets from PCAP")

        perturbed_packets = []
        for pkt in packets:
            result = self._process_packet(pkt)
            if result is not None:
                perturbed_packets.append(result)

        # Write output PCAP
        out_path = pcap_file.parent / f"{pcap_file.stem}_perturbed{pcap_file.suffix}"
        self._injector.write_pcap(perturbed_packets, str(out_path))

        self.logger.info(
            "Demo complete",
            output=str(out_path),
            **self.metrics.to_dict(),
        )
        return {"output_pcap": str(out_path), "metrics": self.metrics.to_dict()}

    def run_live(self) -> None:
        """Run the engine in live mode: sniff → perturb → inject continuously."""
        self._init_components()
        self._running.set()

        self.logger.info(
            "Starting live capture mode",
            interface=self.config.engine.interface,
        )

        # Start worker threads
        workers = []
        for i in range(self.config.engine.workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"ape-worker-{i}", daemon=True
            )
            t.start()
            workers.append(t)

        # Start sniffer (blocks; puts packets into queue)
        try:
            self._sniffer.start(callback=self._packet_queue.put_nowait)
        except KeyboardInterrupt:
            self.logger.info("Shutting down engine...")
        finally:
            self._running.clear()

    def _worker_loop(self) -> None:
        """Worker thread: dequeue packets and process them."""
        while self._running.is_set():
            try:
                pkt = self._packet_queue.get(timeout=0.1)
                self._process_packet(pkt)
                self._packet_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("Worker error", error=str(e))
                self.metrics.errors += 1

    def _process_packet(self, raw_packet: Any) -> Optional[Any]:
        """
        Full pipeline for a single packet:
        capture → features → perturb → validate → inject.
        """
        start_time = time.perf_counter()
        self.metrics.packets_captured += 1

        try:
            # 1. Extract features
            features = self._preprocessor.extract(raw_packet)
            if features is None:
                return None

            # 2. Generate adversarial perturbation
            perturbed_features, delta = self._perturber.generate(
                features, self._models
            )
            self.metrics.packets_perturbed += 1

            # 3. Compute MSE distortion
            mse = float(
                ((features - perturbed_features) ** 2).mean()
            )
            self.metrics.update_mse(mse)

            if mse > self.config.perturbation.mse_threshold:
                self.logger.debug("MSE exceeded threshold, skipping injection", mse=mse)
                return None

            # 4. Apply perturbation back to packet
            perturbed_packet = self._injector.apply(raw_packet, perturbed_features)
            self.metrics.packets_injected += 1

            # Track latency
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.update_latency(elapsed_ms)

            if elapsed_ms > self.config.engine.max_latency_ms:
                self.logger.warning("Latency budget exceeded", ms=elapsed_ms)

            return perturbed_packet

        except Exception as e:
            self.logger.error("Packet processing error", error=str(e))
            self.metrics.errors += 1
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Return current runtime metrics as a dict."""
        return self.metrics.to_dict()

    def stop(self) -> None:
        """Signal the engine to stop."""
        self._running.clear()
        self.logger.info("Engine stopped", **self.metrics.to_dict())


# ─── CLI Entry Point ────────────────────────────────────────────────────────

@click.group()
@click.option("--config", "-c", default=None, help="Path to YAML config file")
@click.option("--epsilon", "-e", default=None, type=float, help="Perturbation epsilon")
@click.option("--method", "-m", default=None, type=click.Choice(["fgsm", "pgd", "ensemble"]))
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], epsilon: Optional[float], method: Optional[str]):
    """APE-BAIT: Adversarial Perturbation Engine to Blind Attacker AI Tools."""
    overrides: Dict[str, Any] = {}
    if epsilon is not None:
        overrides.setdefault("perturbation", {})["epsilon"] = epsilon
    if method is not None:
        overrides.setdefault("perturbation", {})["method"] = method

    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path=config, overrides=overrides)


@cli.command()
@click.option("--pcap", "-p", required=True, help="Path to input PCAP file")
@click.pass_context
def demo(ctx: click.Context, pcap: str):
    """Run APE-BAIT in demo mode (PCAP replay)."""
    cfg: APEBAITConfig = ctx.obj["config"]
    engine = APEBAITEngine(cfg)
    results = engine.run_demo(pcap)
    click.echo(f"\n✅ Demo complete!")
    click.echo(f"   Output PCAP : {results['output_pcap']}")
    click.echo(f"   Metrics     : {results['metrics']}")


@cli.command()
@click.pass_context
def live(ctx: click.Context):
    """Run APE-BAIT in live network capture mode."""
    cfg: APEBAITConfig = ctx.obj["config"]
    engine = APEBAITEngine(cfg)
    engine.run_live()


if __name__ == "__main__":
    cli()
