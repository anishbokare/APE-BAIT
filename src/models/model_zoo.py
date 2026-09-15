"""
APE-BAIT Model Zoo
Registry and lifecycle manager for all surrogate models.
Handles loading, caching, device placement, and eval mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.core.config import APEBAITConfig
from src.core.logger import get_logger
from src.models.surrogate_ids import SurrogateIDS, build_surrogate_ids
from src.models.surrogate_malware import SurrogateMalwareClassifier, build_surrogate_malware
from src.models.surrogate_anomaly import SurrogateAnomalyDetector, build_surrogate_anomaly


class ModelZoo:
    """
    Central registry for all APE-BAIT surrogate models.

    Provides:
      - Lazy loading with caching
      - Device management (CPU / CUDA)
      - Consistent eval mode (no dropout during inference)
      - Gradient enablement (requires_grad for adversarial computation)
    """

    def __init__(self, config: APEBAITConfig):
        self.config = config
        self.device = torch.device(config.models.device)
        self.logger = get_logger("ape-bait.models")
        self._cache: Dict[str, nn.Module] = {}

    # ─── Accessors ──────────────────────────────────────────────

    @property
    def ids(self) -> SurrogateIDS:
        """Get (or load) the IDS surrogate model."""
        return self._get_model(
            "ids",
            build_fn=lambda: build_surrogate_ids(
                input_dim=self.config.models.feature_dim,
                pretrained_path=self._resolve_path(self.config.models.paths.ids),
                device=str(self.device),
            ),
        )

    @property
    def malware(self) -> SurrogateMalwareClassifier:
        """Get (or load) the malware classifier surrogate."""
        return self._get_model(
            "malware",
            build_fn=lambda: build_surrogate_malware(
                input_dim=self.config.models.feature_dim,
                pretrained_path=self._resolve_path(self.config.models.paths.malware),
                device=str(self.device),
            ),
        )

    @property
    def anomaly(self) -> SurrogateAnomalyDetector:
        """Get (or load) the anomaly detector surrogate."""
        return self._get_model(
            "anomaly",
            build_fn=lambda: build_surrogate_anomaly(
                input_dim=self.config.models.feature_dim,
                pretrained_path=self._resolve_path(self.config.models.paths.anomaly),
                device=str(self.device),
            ),
        )

    def all_models(self) -> Dict[str, nn.Module]:
        """Return dict of all loaded surrogate models."""
        return {
            "ids": self.ids,
            "malware": self.malware,
            "anomaly": self.anomaly,
        }

    def preload_all(self) -> None:
        """Force-load all models into memory."""
        self.logger.info("Preloading all surrogate models...")
        _ = self.ids
        _ = self.malware
        _ = self.anomaly
        self.logger.info(
            f"Models loaded on device: {self.device}",
            models=list(self._cache.keys()),
        )

    def save_model(self, name: str, path: str) -> None:
        """Save a loaded model's state dict."""
        if name not in self._cache:
            raise KeyError(f"Model '{name}' not loaded yet.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._cache[name].state_dict(), path)
        self.logger.info(f"Saved model '{name}' to {path}")

    def save_all(self) -> None:
        """Save all cached models to their configured paths."""
        path_map = {
            "ids": self.config.models.paths.ids,
            "malware": self.config.models.paths.malware,
            "anomaly": self.config.models.paths.anomaly,
        }
        for name, path in path_map.items():
            if name in self._cache:
                self.save_model(name, path)

    # ─── Internal ───────────────────────────────────────────────

    def _get_model(self, name: str, build_fn) -> nn.Module:
        """Get cached model or build and cache it."""
        if name not in self._cache:
            self.logger.info(f"Loading surrogate model: {name}")
            model = build_fn()
            model.eval()   # Always eval mode (BN/Dropout in inference mode)
            self._cache[name] = model
        return self._cache[name]

    def _resolve_path(self, path_str: str) -> Optional[str]:
        """Resolve model weight path; return None if file doesn't exist."""
        p = Path(path_str)
        if p.exists():
            return str(p)
        # Resolve relative to project root
        root = Path(__file__).parent.parent.parent
        full = root / path_str
        if full.exists():
            return str(full)
        # No weights found — model will use random init
        self.logger.warning(f"Model weights not found at {path_str} — using random init")
        return None

    def to_tensor(self, features, requires_grad: bool = True) -> torch.Tensor:
        """Convert numpy features to tensor on the right device with grad."""
        import numpy as np
        if isinstance(features, np.ndarray):
            t = torch.from_numpy(features.copy()).float()
        elif isinstance(features, torch.Tensor):
            t = features.float()
        else:
            t = torch.tensor(features, dtype=torch.float32)
        t = t.to(self.device)
        if requires_grad:
            t.requires_grad_(True)
        return t
