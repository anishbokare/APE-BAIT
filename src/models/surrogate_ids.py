"""
APE-BAIT Surrogate IDS Model
CNN-based differentiable surrogate approximating IDS tools like
Snort/Suricata ML plugins. Trained on CICIDS-style features.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateIDS(nn.Module):
    """
    1D CNN surrogate model for Intrusion Detection Systems.

    Architecture:
      Input (batch, feature_dim) →
        Linear embedding →
        Reshape to (batch, 1, feature_dim) →
        3x Conv1d blocks (with BN + ReLU) →
        Adaptive avg pool →
        FC classifier →
        Binary output [benign=0, malicious=1]
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Initial projection
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # → (batch, 128, 1)
            nn.Flatten(),              # → (batch, 128)
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Project & reshape for conv
        x = self.input_proj(x)                    # (B, input_dim)
        x = x.unsqueeze(1)                         # (B, 1, input_dim)
        x = self.conv_blocks(x)                    # (B, 128, input_dim)
        x = self.classifier(x)                     # (B, num_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (0=benign, 1=malicious)."""
        return self.predict_proba(x).argmax(dim=-1)

    @property
    def loss_fn(self):
        return nn.CrossEntropyLoss()


def build_surrogate_ids(
    input_dim: int = 256,
    num_classes: int = 2,
    pretrained_path: Optional[str] = None,
    device: str = "cpu",
) -> SurrogateIDS:
    """
    Build and optionally load pre-trained SurrogateIDS model.

    Args:
        input_dim: Feature vector dimension.
        num_classes: Number of output classes.
        pretrained_path: Path to .pt weights file.
        device: torch device string.

    Returns:
        SurrogateIDS model ready for inference/gradient computation.
    """
    model = SurrogateIDS(input_dim=input_dim, num_classes=num_classes)

    if pretrained_path:
        try:
            state = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state)
            print(f"[SurrogateIDS] Loaded weights from {pretrained_path}")
        except Exception as e:
            print(f"[SurrogateIDS] Could not load weights: {e}. Using random init.")

    model.to(device)
    return model
