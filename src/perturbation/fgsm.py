"""
APE-BAIT FGSM Perturbation Generator
Fast Gradient Sign Method — single-step adversarial attack.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
           ICLR 2015. https://arxiv.org/abs/1412.6572

Formula:
    x_adv = x + ε · sign(∇_x L(θ, x, y))

For evasion (push towards benign class):
    x_adv = x - ε · sign(∇_x L(θ, x, y_malicious))
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


class FGSMPerturbation(BasePerturbationGenerator):
    """
    Fast Gradient Sign Method (FGSM) perturbation generator.

    Single-step attack: fastest latency, lower evasion than PGD.
    Ideal for high-throughput scenarios where latency budget is tight.

    Characteristics:
      - Latency: ~1-5ms per packet (CPU)
      - Evasion rate: ~70-80% (vs PGD's 80-90%)
      - MSE: predictable, bounded by ε²
    """

    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.epsilon = config.fgsm.epsilon
        self.targeted = config.fgsm.targeted
        self.logger = get_logger("ape-bait.fgsm")
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
        Generate FGSM adversarial perturbation.

        Args:
            features: Feature vector (feature_dim,), float32.
            model_zoo: ModelZoo with surrogate models.

        Returns:
            (perturbed_features, delta) both shape (feature_dim,).
        """
        start = time.perf_counter()

        # Use IDS surrogate as primary gradient source
        model = model_zoo.ids
        device = model_zoo.device

        # Convert to tensor with grad
        x = torch.from_numpy(features.copy()).float().unsqueeze(0).to(device)
        x.requires_grad_(True)

        # Forward pass
        model.eval()
        output = model(x)

        # Loss: push towards benign class (0)
        if self.targeted:
            target = torch.zeros(1, dtype=torch.long, device=device)  # class 0 = benign
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            # Untargeted: maximize loss for malicious class (1)
            target = torch.ones(1, dtype=torch.long, device=device)
            loss = -nn.CrossEntropyLoss()(output, target)

        # Compute gradient
        loss.backward()
        grad = x.grad.data  # (1, feature_dim)

        # FGSM step: x_adv = x - ε * sign(∇L)  [negate because we minimize loss]
        sign_grad = grad.sign()
        x_adv = x + self.epsilon * sign_grad  # Push towards benign

        # Project back to valid range
        x_adv_proj = self.constraints.project_linf(x_adv, x.detach())

        perturbed = x_adv_proj.squeeze(0).detach().cpu().numpy().astype(np.float32)
        delta = perturbed - features

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.logger.debug(
            "FGSM complete",
            latency_ms=round(elapsed_ms, 2),
            mse=round(float(np.mean(delta ** 2)), 6),
            linf=round(float(np.max(np.abs(delta))), 6),
        )

        return perturbed, delta

    def generate_batch(
        self,
        features_batch: np.ndarray,
        model_zoo: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched FGSM — more efficient than element-wise for GPU.

        Args:
            features_batch: (N, feature_dim) float32
            model_zoo: ModelZoo

        Returns:
            (perturbed_batch, deltas) each (N, feature_dim)
        """
        model = model_zoo.ids
        device = model_zoo.device

        x = torch.from_numpy(features_batch.copy()).float().to(device)
        x.requires_grad_(True)

        model.eval()
        output = model(x)
        N = x.shape[0]

        if self.targeted:
            target = torch.zeros(N, dtype=torch.long, device=device)
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            target = torch.ones(N, dtype=torch.long, device=device)
            loss = -nn.CrossEntropyLoss()(output, target)

        loss.backward()
        sign_grad = x.grad.data.sign()
        x_adv = x + self.epsilon * sign_grad

        perturbed_list = []
        x_orig = torch.from_numpy(features_batch).float().to(device)
        for i in range(N):
            x_adv_proj = self.constraints.project_linf(
                x_adv[i:i+1], x_orig[i:i+1]
            )
            perturbed_list.append(x_adv_proj.squeeze(0).detach().cpu().numpy())

        perturbed_batch = np.stack(perturbed_list).astype(np.float32)
        deltas = perturbed_batch - features_batch
        return perturbed_batch, deltas
