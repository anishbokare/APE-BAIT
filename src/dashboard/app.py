"""
APE-BAIT Real-Time Monitoring Dashboard
Flask + Flask-SocketIO web dashboard for live engine monitoring.

Features:
  - Real-time metrics streaming via WebSocket
  - Evasion rate time-series chart
  - Latency histogram
  - MSE gauge
  - Tool-by-tool evasion breakdown
  - Engine control panel (start/stop/configure)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import click
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from src.core.config import APEBAITConfig, load_config
from src.core.logger import get_logger


# ─── Flask App Setup ────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder="../../templates",
    static_folder="../../static",
)
app.config["SECRET_KEY"] = "ape-bait-secret-key-change-in-production"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

logger = get_logger("ape-bait.dashboard")

# Global engine reference (set via init_app or CLI)
_engine: Optional[Any] = None
_config: Optional[APEBAITConfig] = None
_push_thread: Optional[threading.Thread] = None
_push_active = threading.Event()


# ─── Simulated Demo Metrics ──────────────────────────────────────────────────

class DemoMetricsSimulator:
    """
    Generates realistic-looking demo metrics when no live engine is running.
    Used when the dashboard is opened without a connected APE-BAIT engine.
    """

    def __init__(self):
        import random
        self._rng = random.Random(42)
        self._t = 0
        self._packets = 0
        self._base_evasion = 0.82

    def next(self) -> Dict[str, Any]:
        import random
        self._t += 1
        self._packets += self._rng.randint(30, 80)
        evasion = min(0.98, self._base_evasion + self._rng.gauss(0, 0.03))
        latency = 850 + self._rng.gauss(0, 120)
        mse = 0.018 + self._rng.gauss(0, 0.003)

        return {
            "packets_captured": self._packets,
            "packets_perturbed": int(self._packets * 0.97),
            "packets_injected": int(self._packets * 0.95),
            "evasion_rate": round(max(0, min(1, evasion)), 4),
            "avg_latency_ms": round(max(10, latency), 1),
            "avg_mse": round(max(0.001, mse), 6),
            "throughput_pps": round(self._rng.uniform(800, 1200), 1),
            "errors": self._rng.randint(0, 2),
            "uptime_seconds": self._t * 0.5,
            "method": "pgd",
            "epsilon": 0.03,
            "tool_evasion": {
                "Snort": round(0.85 + self._rng.gauss(0, 0.02), 3),
                "Suricata": round(0.83 + self._rng.gauss(0, 0.02), 3),
                "YARA": round(0.91 + self._rng.gauss(0, 0.02), 3),
                "ClamAV": round(0.78 + self._rng.gauss(0, 0.02), 3),
                "Zeek": round(0.82 + self._rng.gauss(0, 0.02), 3),
                "MalConv": round(0.88 + self._rng.gauss(0, 0.02), 3),
                "IForest": round(0.79 + self._rng.gauss(0, 0.02), 3),
                "CNN IDS": round(0.84 + self._rng.gauss(0, 0.02), 3),
            },
        }


_demo_sim = DemoMetricsSimulator()


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/metrics")
def api_metrics():
    """REST endpoint: get current engine metrics."""
    if _engine is not None:
        data = _engine.get_metrics()
    else:
        data = _demo_sim.next()
    return jsonify(data)


@app.route("/api/status")
def api_status():
    """REST endpoint: get engine status."""
    return jsonify({
        "status": "running" if _engine else "demo",
        "version": "1.0.0",
        "config": {
            "method": _config.perturbation.method if _config else "pgd",
            "epsilon": _config.perturbation.epsilon if _config else 0.03,
            "device": _config.models.device if _config else "cpu",
        } if _config else {},
    })


@app.route("/api/control", methods=["POST"])
def api_control():
    """REST endpoint: engine control commands."""
    data = request.get_json() or {}
    action = data.get("action")

    if action == "stop" and _engine:
        _engine.stop()
        return jsonify({"status": "stopped"})
    elif action == "status":
        return api_status()
    else:
        return jsonify({"status": "unknown_action", "action": action}), 400


@app.route("/api/validation")
def api_validation():
    """Return simulated validation results for the dashboard."""
    return jsonify({
        "tools": [
            {"name": "Snort", "evasion_rate": 0.85, "original": 47, "perturbed": 7, "type": "Rule-based IDS"},
            {"name": "Suricata", "evasion_rate": 0.83, "original": 41, "perturbed": 7, "type": "Rule-based IDS"},
            {"name": "YARA", "evasion_rate": 0.91, "original": 22, "perturbed": 2, "type": "Pattern Matching"},
            {"name": "ClamAV", "evasion_rate": 0.78, "original": 18, "perturbed": 4, "type": "Signature AV"},
            {"name": "Zeek", "evasion_rate": 0.82, "original": 33, "perturbed": 6, "type": "Network Monitor"},
            {"name": "MalConv", "evasion_rate": 0.88, "original": 25, "perturbed": 3, "type": "Deep Learning"},
            {"name": "CNN IDS", "evasion_rate": 0.84, "original": 31, "perturbed": 5, "type": "ML-based IDS"},
            {"name": "IForest", "evasion_rate": 0.79, "original": 28, "perturbed": 6, "type": "Anomaly"},
            {"name": "LOF", "evasion_rate": 0.76, "original": 25, "perturbed": 6, "type": "Anomaly"},
            {"name": "AE Detector", "evasion_rate": 0.81, "original": 26, "perturbed": 5, "type": "Anomaly"},
            {"name": "Ember LGBM", "evasion_rate": 0.80, "original": 30, "perturbed": 6, "type": "ML Malware"},
            {"name": "DGA Detector", "evasion_rate": 0.73, "original": 15, "perturbed": 4, "type": "DNS"},
            {"name": "Port Scan Det.", "evasion_rate": 0.68, "original": 22, "perturbed": 7, "type": "Heuristic"},
            {"name": "RF IDS", "evasion_rate": 0.86, "original": 36, "perturbed": 5, "type": "ML-based IDS"},
            {"name": "LSTM IDS", "evasion_rate": 0.82, "original": 27, "perturbed": 5, "type": "ML-based IDS"},
        ],
        "aggregate": {
            "evasion_rate": 0.808,
            "avg_mse": 0.018,
            "avg_latency_ms": 847,
        }
    })


# ─── WebSocket Events ────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    """Client connected — start or confirm push thread."""
    logger.info("Dashboard client connected")
    emit("status", {"connected": True, "mode": "demo" if _engine is None else "live"})
    _ensure_push_thread()


@socketio.on("disconnect")
def on_disconnect():
    logger.info("Dashboard client disconnected")


@socketio.on("request_metrics")
def on_request_metrics():
    """Manual metrics request from client."""
    if _engine:
        emit("metrics", _engine.get_metrics())
    else:
        emit("metrics", _demo_sim.next())


def _ensure_push_thread():
    """Start background metrics push thread if not already running."""
    global _push_thread
    if _push_thread and _push_thread.is_alive():
        return
    _push_active.set()
    _push_thread = threading.Thread(
        target=_metrics_push_loop, daemon=True, name="metrics-push"
    )
    _push_thread.start()


def _metrics_push_loop():
    """Background thread: push metrics to all WebSocket clients every 500ms."""
    while _push_active.is_set():
        try:
            if _engine:
                data = _engine.get_metrics()
            else:
                data = _demo_sim.next()
            socketio.emit("metrics", data, namespace="/")
        except Exception as e:
            logger.debug(f"Push error: {e}")
        time.sleep(0.5)


# ─── App Initialization ───────────────────────────────────────────────────────

def init_app(engine=None, config: Optional[APEBAITConfig] = None):
    """Initialize the dashboard with an optional engine reference."""
    global _engine, _config
    _engine = engine
    _config = config


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

@click.command()
@click.option("--host", default="0.0.0.0", help="Dashboard host")
@click.option("--port", default=8080, help="Dashboard port")
@click.option("--config", "-c", default=None, help="APE-BAIT config file")
@click.option("--debug", is_flag=True, default=False, help="Debug mode")
def cli(host: str, port: int, config: Optional[str], debug: bool):
    """Launch the APE-BAIT real-time monitoring dashboard."""
    global _config
    _config = load_config(config_path=config)

    logger.info(
        f"Starting APE-BAIT Dashboard",
        host=host,
        port=port,
        debug=debug,
    )
    print(f"\nAPE-BAIT Dashboard running at http://{host}:{port}\n")
    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    cli()
