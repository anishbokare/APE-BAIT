"""
APE-BAIT Base Perturbation Generator
Abstract interface for all adversarial perturbation strategies.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.core.config import PerturbationConfig


class BasePerturbationGenerator(abc.ABC):
    """
    Abstract base class for adversarial perturbation generators.

    All concrete implementations (FGSM, PGD, Ensemble) must implement
    the `generate` method.

    The generator computes gradients against a surrogate model and
    produces an adversarial perturbation delta constrained by:
      - L∞ norm bound (epsilon)
      - MSE distortion threshold
      - Byte-validity constraints (for packet features)
    """

    def __init__(self, config: PerturbationConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.mse_threshold = config.mse_threshold

    @abc.abstractmethod
    def generate(
        self,
        features: np.ndarray,
        model_zoo: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adversarial perturbation for a single feature vector.

        Args:
            features: Original feature array of shape (feature_dim,), float32.
            model_zoo: ModelZoo instance providing access to surrogate models.

        Returns:
            Tuple of:
              - perturbed_features: Adversarial feature array (feature_dim,), float32
              - delta: Perturbation delta (feature_dim,), float32
        """
        ...

    def generate_batch(
        self,
        features_batch: np.ndarray,
        model_zoo: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbations for a batch of feature vectors.

        Default implementation: apply generate() element-wise.
        Subclasses should override for batched gradient computation.

        Args:
            features_batch: Feature array of shape (N, feature_dim), float32.
            model_zoo: ModelZoo instance.

        Returns:
            Tuple of (perturbed_batch, deltas) each of shape (N, feature_dim).
        """
        perturbed_list, delta_list = [], []
        for feat in features_batch:
            p, d = self.generate(feat, model_zoo)
            perturbed_list.append(p)
            delta_list.append(d)
        return np.stack(perturbed_list), np.stack(delta_list)

    def _compute_loss_and_grad(
        self,
        x_tensor: torch.Tensor,
        model: nn.Module,
        targeted: bool = False,
        target_class: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and gradient for a given model.

        For untargeted attack: maximize loss w.r.t. true class (class 1 = malicious).
        For targeted attack: minimize loss w.r.t. target class.

        Returns:
            Tuple of (loss scalar, gradient tensor)
        """
        if x_tensor.grad is not None:
            x_tensor.grad.zero_()

        # Ensure grad is tracked
        x_tensor.requires_grad_(True)

        output = model(x_tensor)

        if hasattr(model, 'reconstruction_error'):
            # Anomaly detector: minimize reconstruction error
            loss = model.reconstruction_error(x_tensor).mean()
        elif output.shape[-1] > 1:
            # Classifier: push towards benign class (0)
            if targeted:
                loss = nn.CrossEntropyLoss()(
                    output,
                    torch.full((output.shape[0],), target_class, dtype=torch.long, device=output.device)
                )
            else:
                # Untargeted: maximize loss for class 1 (malicious)
                # i.e., make model think it's benign (class 0)
                true_labels = torch.ones(
                    output.shape[0], dtype=torch.long, device=output.device
                )
                loss = -nn.CrossEntropyLoss()(output, true_labels)
        else:
            loss = output.mean()

        loss.backward()
        grad = x_tensor.grad.clone()
        return loss, grad

    def _apply_linf_clip(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
    ) -> torch.Tensor:
        """Project x_adv back into the L∞ epsilon-ball around x_orig."""
        delta = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
        return torch.clamp(x_orig + delta, 0.0, 1.0)

    def _validate_mse(self, original: np.ndarray, perturbed: np.ndarray) -> bool:
        """Check whether perturbation satisfies the MSE stealth constraint."""
        mse = float(np.mean((original - perturbed) ** 2))
        return mse <= self.mse_threshold

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)
