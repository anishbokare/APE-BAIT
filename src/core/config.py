"""
APE-BAIT Configuration Loader
Handles YAML config loading, validation, and runtime overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ─── Sub-config dataclasses ─────────────────────────────────────────────────

@dataclass
class FGSMConfig:
    epsilon: float = 0.03
    targeted: bool = False


@dataclass
class PGDConfig:
    steps: int = 40
    step_size: float = 0.003
    random_start: bool = True
    restarts: int = 3
    early_stop: bool = True


@dataclass
class EnsembleWeights:
    ids: float = 0.4
    malware: float = 0.4
    anomaly: float = 0.2


@dataclass
class PerturbationConfig:
    method: str = "pgd"          # fgsm | pgd | ensemble
    epsilon: float = 0.03
    mse_threshold: float = 0.04
    fgsm: FGSMConfig = field(default_factory=FGSMConfig)
    pgd: PGDConfig = field(default_factory=PGDConfig)
    ensemble: EnsembleWeights = field(default_factory=EnsembleWeights)


@dataclass
class ModelPaths:
    ids: str = "data/models/surrogate_ids.pt"
    malware: str = "data/models/surrogate_malware.pt"
    anomaly: str = "data/models/surrogate_anomaly.pt"


@dataclass
class ModelsConfig:
    device: str = "cpu"
    paths: ModelPaths = field(default_factory=ModelPaths)
    feature_dim: int = 256
    payload_max_len: int = 512


@dataclass
class CaptureConfig:
    bpf_filter: str = ""
    snapshot_len: int = 65535
    promisc: bool = True
    timeout_ms: int = 100


@dataclass
class InjectionConfig:
    mode: str = "replay"
    recalc_checksums: bool = True
    rate_limit_pps: int = 10000


@dataclass
class EngineConfig:
    mode: str = "demo"
    interface: str = "eth0"
    batch_size: int = 32
    max_latency_ms: int = 2000
    queue_size: int = 1024
    workers: int = 4
    log_level: str = "INFO"


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    update_interval_ms: int = 500


@dataclass
class ValidationConfig:
    tools: List[str] = field(default_factory=lambda: [
        "snort", "suricata", "yara", "clamav", "zeek",
        "ml_ids_cnn", "ml_ids_rf", "ml_ids_lstm",
        "anomaly_iforest", "anomaly_lof", "anomaly_autoencoder",
        "malconv", "ember_lgbm", "dga_detector", "port_scan_detector"
    ])
    parallel: bool = True
    timeout_per_tool: int = 60


# ─── Root Config ────────────────────────────────────────────────────────────

@dataclass
class APEBAITConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ─── Loader ─────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts; override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _from_dict(dataclass_type, data: Dict[str, Any]):
    """Recursively instantiate a dataclass from a dict."""
    import dataclasses
    if not dataclasses.is_dataclass(dataclass_type):
        return data
    kwargs = {}
    for f in dataclasses.fields(dataclass_type):
        if f.name in data:
            val = data[f.name]
            if dataclasses.is_dataclass(f.type) or (
                isinstance(f.type, type) and dataclasses.is_dataclass(f.type)
            ):
                val = _from_dict(f.type, val) if isinstance(val, dict) else val
            elif hasattr(f, 'default_factory'):
                # Try to resolve type annotation for nested dataclasses
                origin_type = f.type if isinstance(f.type, type) else None
                if origin_type and dataclasses.is_dataclass(origin_type) and isinstance(val, dict):
                    val = _from_dict(origin_type, val)
            kwargs[f.name] = val
    return dataclass_type(**kwargs)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> APEBAITConfig:
    """
    Load APE-BAIT configuration.

    Args:
        config_path: Path to YAML config file. Defaults to config/default.yaml.
        overrides: Dict of values to override loaded config.

    Returns:
        Populated APEBAITConfig dataclass.
    """
    # Find default config
    default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    raw: Dict[str, Any] = {}

    # Load default config
    if default_path.exists():
        with open(default_path, "r") as f:
            raw = yaml.safe_load(f) or {}

    # Load user-supplied config on top
    if config_path:
        user_path = Path(config_path)
        if not user_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(user_path, "r") as f:
            user_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, user_raw)

    # Apply runtime overrides
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Apply environment variable overrides
    # APE_BAIT_DEVICE=cuda → models.device=cuda
    env_map = {
        "APE_BAIT_DEVICE": ["models", "device"],
        "APE_BAIT_EPSILON": ["perturbation", "epsilon"],
        "APE_BAIT_METHOD": ["perturbation", "method"],
        "APE_BAIT_INTERFACE": ["engine", "interface"],
        "APE_BAIT_PORT": ["dashboard", "port"],
    }
    for env_key, config_path_parts in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            d = raw
            for part in config_path_parts[:-1]:
                d = d.setdefault(part, {})
            d[config_path_parts[-1]] = val

    # Build nested dataclasses
    cfg = APEBAITConfig(
        engine=_from_dict(EngineConfig, raw.get("engine", {})),
        perturbation=_build_perturbation_config(raw.get("perturbation", {})),
        models=_build_models_config(raw.get("models", {})),
        capture=_from_dict(CaptureConfig, raw.get("capture", {})),
        injection=_from_dict(InjectionConfig, raw.get("injection", {})),
        dashboard=_from_dict(DashboardConfig, raw.get("dashboard", {})),
        validation=_from_dict(ValidationConfig, raw.get("validation", {})),
    )
    return cfg


def _build_perturbation_config(data: Dict[str, Any]) -> PerturbationConfig:
    return PerturbationConfig(
        method=data.get("method", "pgd"),
        epsilon=float(data.get("epsilon", 0.03)),
        mse_threshold=float(data.get("mse_threshold", 0.04)),
        fgsm=_from_dict(FGSMConfig, data.get("fgsm", {})),
        pgd=_from_dict(PGDConfig, data.get("pgd", {})),
        ensemble=_from_dict(EnsembleWeights, data.get("ensemble", {}).get("weights", {})),
    )


def _build_models_config(data: Dict[str, Any]) -> ModelsConfig:
    paths_data = data.get("paths", {})
    return ModelsConfig(
        device=data.get("device", "cpu"),
        paths=_from_dict(ModelPaths, paths_data),
        feature_dim=int(data.get("feature_dim", 256)),
        payload_max_len=int(data.get("payload_max_len", 512)),
    )
