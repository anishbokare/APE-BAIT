"""
APE-BAIT Surrogate Anomaly Detector
Autoencoder-based anomaly detector surrogate that mimics
behavioral anomaly detection systems (Isolation Forest, LOF,
LSTM-Autoencoder-based detectors).

The surrogate is trained to reconstruct normal traffic features.
High reconstruction error → anomaly flag.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateAnomalyDetector(nn.Module):
    """
    Autoencoder-based anomaly detector surrogate.

    Architecture:
      Encoder: input_dim → 128 → 64 → latent_dim
      Decoder: latent_dim → 64 → 128 → input_dim

    Anomaly score = MSE reconstruction error.
    Perturbations aim to REDUCE the reconstruction error
    (make malicious traffic look normal to the autoencoder).
    """

    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (128, 64),
        threshold: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold = threshold

        # ── Encoder ─────────────────────────────────────────────
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Decoder ─────────────────────────────────────────────
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_dim = h_dim
        decoder_layers += [
            nn.Linear(in_dim, input_dim),
            nn.Sigmoid(),   # Output in [0, 1] (normalized features)
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, input_dim)

        Returns:
            Tuple of (reconstruction, latent_code)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error (anomaly score).

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Reconstruction MSE per sample (batch,)
        """
        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def is_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Binary anomaly detection decision.

        Returns:
            Boolean tensor (batch,): True if anomaly detected.
        """
        scores = self.reconstruction_error(x)
        return scores > self.threshold

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw anomaly scores (higher = more anomalous).
        Used as the loss target for adversarial perturbation
        (minimize this to make traffic appear normal).
        """
        return self.reconstruction_error(x)

    @property
    def loss_fn(self):
        """Loss for adversarial: minimize reconstruction error."""
        return nn.MSELoss()


def build_surrogate_anomaly(
    input_dim: int = 256,
    latent_dim: int = 32,
    threshold: float = 0.05,
    pretrained_path: Optional[str] = None,
    device: str = "cpu",
) -> SurrogateAnomalyDetector:
    """Build and optionally load SurrogateAnomalyDetector."""
    model = SurrogateAnomalyDetector(
        input_dim=input_dim,
        latent_dim=latent_dim,
        threshold=threshold,
    )

    if pretrained_path:
        try:
            state = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state)
            print(f"[SurrogateAnomaly] Loaded weights from {pretrained_path}")
        except Exception as e:
            print(f"[SurrogateAnomaly] Could not load weights: {e}. Using random init.")

    model.to(device)
    return model
