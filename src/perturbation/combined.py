"""
APE-BAIT Ensemble Perturbation Generator
Combines gradients from multiple surrogate models (IDS + Malware + Anomaly)
for stronger, more transferable adversarial perturbations.

Transfer attack strategy: perturbations computed against an ensemble of
surrogates generalize better to unseen black-box target models.

Loss formulation:
    L_total = w_ids * L_ids + w_malware * L_malware + w_anomaly * L_anomaly
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.core.config import PerturbationConfig
from src.core.logger import get_logger
from src.perturbation.base import BasePerturbationGenerator
from src.perturbation.constraints import PerturbationConstraints


class EnsemblePerturbation(BasePerturbationGenerator):
    """
    Ensemble perturbation generator using combined surrogate model gradients.

    Generates adversarial examples that fool IDS + malware classifiers +
    anomaly detectors simultaneously, using a weighted combination of losses.

    Combines PGD-style iteration with ensemble gradient computation for
    maximum transfer attack effectiveness.
    """

    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.epsilon = config.epsilon
        # PGD-style iteration for ensemble
        self.steps = config.pgd.steps
        self.step_size = config.pgd.step_size
        self.random_start = config.pgd.random_start
        self.early_stop = config.pgd.early_stop
        # Ensemble weights
        self.w_ids = config.ensemble.ids
        self.w_malware = config.ensemble.malware
        self.w_anomaly = config.ensemble.anomaly
        self.logger = get_logger("ape-bait.ensemble")
        self.constraints = PerturbationConstraints(
            epsilon=self.epsilon,
            mse_threshold=config.mse_threshold,
        )

    def generate(
        self,
        features: np.ndarray,
        model_zoo: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble adversarial perturbation via multi-model gradient combination.

        Args:
            features: Feature vector (feature_dim,), float32.
            model_zoo: ModelZoo with all surrogate models.

        Returns:
            (perturbed_features, delta) both shape (feature_dim,).
        """
        start = time.perf_counter()
        device = model_zoo.device

        x_orig = torch.from_numpy(features.copy()).float().unsqueeze(0).to(device)

        # Random start
        if self.random_start:
            noise = torch.zeros_like(x_orig).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_orig + noise, 0.0, 1.0).detach()
        else:
            x_adv = x_orig.clone().detach()

        ids_model = model_zoo.ids
        malware_model = model_zoo.malware
        anomaly_model = model_zoo.anomaly

        for step in range(self.steps):
            x_adv.requires_grad_(True)

            # Compute combined ensemble loss
            loss = self._ensemble_loss(x_adv, ids_model, malware_model, anomaly_model, device)
            loss.backward()
            grad = x_adv.grad.data

            with torch.no_grad():
                x_adv = x_adv + self.step_size * grad.sign()
                x_adv = self.constraints.project_linf(x_adv, x_orig)

            # Early stop: check all models agree it's benign
            if self.early_stop and step % 5 == 0:
                if self._check_evasion(x_adv, ids_model, malware_model, device):
                    self.logger.debug(f"Ensemble: evasion achieved at step {step + 1}")
                    break

        perturbed = x_adv.squeeze(0).detach().cpu().numpy().astype(np.float32)
        delta = perturbed - features

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.logger.debug(
            "Ensemble perturbation complete",
            latency_ms=round(elapsed_ms, 2),
            mse=round(float(np.mean(delta ** 2)), 6),
            linf=round(float(np.max(np.abs(delta))), 6),
        )

        return perturbed, delta

    def _ensemble_loss(
        self,
        x_adv: torch.Tensor,
        ids_model: nn.Module,
        malware_model: nn.Module,
        anomaly_model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute weighted combination of all surrogate losses."""
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)

        # IDS surrogate: push towards benign (class 0)
        if self.w_ids > 0:
            out_ids = ids_model(x_adv)
            target_ids = torch.ones(1, dtype=torch.long, device=device)
            loss_ids = -nn.CrossEntropyLoss()(out_ids, target_ids)  # Negate for evasion
            total_loss = total_loss + self.w_ids * loss_ids

        # Malware surrogate: push towards benign class
        if self.w_malware > 0:
            out_mal = malware_model(x_adv)
            # Target class 0 = benign
            target_mal = torch.zeros(1, dtype=torch.long, device=device)
            loss_mal = nn.CrossEntropyLoss()(out_mal, target_mal)
            total_loss = total_loss + self.w_malware * loss_mal

        # Anomaly surrogate: minimize reconstruction error (look normal)
        if self.w_anomaly > 0:
            score = anomaly_model.anomaly_score(x_adv)
            loss_anom = score.mean()  # Minimize anomaly score
            total_loss = total_loss + self.w_anomaly * loss_anom

        return total_loss

    def _check_evasion(
        self,
        x_adv: torch.Tensor,
        ids_model: nn.Module,
        malware_model: nn.Module,
        device: torch.device,
    ) -> bool:
        """Check if current adversarial example achieves evasion on IDS and malware."""
        with torch.no_grad():
            ids_pred = ids_model(x_adv).argmax(dim=-1).item()
            mal_pred = malware_model(x_adv).argmax(dim=-1).item()
        return ids_pred == 0 and mal_pred == 0  # Both say benign
