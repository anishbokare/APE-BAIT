"""
APE-BAIT Perturbation Constraints
Utilities for enforcing validity constraints on adversarial perturbations:
  - Lp norm bounds
  - MSE distortion threshold
  - Protocol-aware byte validity
  - Packet structure preservation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ConstraintViolation:
    """Details about a constraint violation."""
    constraint: str
    value: float
    threshold: float
    satisfied: bool = False

    def __str__(self) -> str:
        status = "✓" if self.satisfied else "✗"
        return f"[{status}] {self.constraint}: {self.value:.6f} (threshold: {self.threshold:.6f})"


class PerturbationConstraints:
    """
    Enforces validity constraints on adversarial perturbations.

    Constraints applied (in order):
      1. L∞ norm bound: max(|delta|) ≤ epsilon
      2. [0, 1] feature clipping (normalized byte values)
      3. MSE distortion ≤ mse_threshold
      4. Protocol header preservation (first 8 feature dims protected)
    """

    # Feature indices that correspond to protocol headers (protected)
    # [0]=pkt_len [1]=ip_proto [2-5]=tcp_flags [6]=ttl [7]=window
    PROTECTED_DIMS = list(range(8))

    def __init__(
        self,
        epsilon: float = 0.03,
        mse_threshold: float = 0.04,
        protect_headers: bool = True,
        clip_range: tuple = (0.0, 1.0),
    ):
        self.epsilon = epsilon
        self.mse_threshold = mse_threshold
        self.protect_headers = protect_headers
        self.clip_min, self.clip_max = clip_range

    # ─── PyTorch Tensor Operations ──────────────────────────────

    def project_linf(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project x_adv back into the L∞ epsilon-ball centered at x_orig.

        Args:
            x_adv: Adversarial tensor
            x_orig: Original tensor

        Returns:
            Projected tensor within epsilon-ball, clipped to [clip_min, clip_max]
        """
        delta = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
        x_proj = torch.clamp(x_orig + delta, self.clip_min, self.clip_max)

        if self.protect_headers:
            x_proj = self._restore_protected(x_proj, x_orig)

        return x_proj

    def project_l2(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
        l2_bound: Optional[float] = None,
    ) -> torch.Tensor:
        """Project perturbation onto L2 ball."""
        bound = l2_bound or (self.epsilon * (x_orig.numel() ** 0.5))
        delta = x_adv - x_orig
        norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        delta_proj = delta * (bound / norm.clamp(min=bound))
        x_proj = torch.clamp(x_orig + delta_proj, self.clip_min, self.clip_max)

        if self.protect_headers:
            x_proj = self._restore_protected(x_proj, x_orig)

        return x_proj

    def _restore_protected(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
    ) -> torch.Tensor:
        """Zero out perturbation in protected (header) dimensions."""
        mask = torch.ones_like(x_adv)
        mask[..., self.PROTECTED_DIMS] = 0.0
        return x_orig * (1 - mask) + x_adv * mask

    # ─── NumPy Operations ───────────────────────────────────────

    def clip_numpy(self, delta: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
        """Apply all constraints to a NumPy delta."""
        # L∞ clip
        delta = np.clip(delta, -self.epsilon, self.epsilon)
        # Feature range clip
        x_adv = np.clip(x_orig + delta, self.clip_min, self.clip_max)
        # Restore protected dims
        if self.protect_headers:
            x_adv[..., self.PROTECTED_DIMS] = x_orig[..., self.PROTECTED_DIMS]
        return x_adv

    # ─── Validation ─────────────────────────────────────────────

    def validate(
        self,
        x_orig: np.ndarray,
        x_adv: np.ndarray,
    ) -> list[ConstraintViolation]:
        """
        Validate all constraints for a perturbation.

        Returns:
            List of ConstraintViolation objects.
        """
        delta = x_adv - x_orig
        violations = []

        # 1. L∞ norm
        linf = float(np.max(np.abs(delta)))
        violations.append(ConstraintViolation(
            "L∞ norm", linf, self.epsilon, linf <= self.epsilon
        ))

        # 2. MSE
        mse = float(np.mean(delta ** 2))
        violations.append(ConstraintViolation(
            "MSE distortion", mse, self.mse_threshold, mse <= self.mse_threshold
        ))

        # 3. Feature range
        in_range = bool(np.all(x_adv >= self.clip_min) and np.all(x_adv <= self.clip_max))
        out_of_range = float(np.sum((x_adv < self.clip_min) | (x_adv > self.clip_max)))
        violations.append(ConstraintViolation(
            "Feature range [0,1]", out_of_range, 0, in_range
        ))

        # 4. Protected dims unchanged
        if self.protect_headers:
            header_diff = float(np.max(np.abs(delta[..., self.PROTECTED_DIMS])))
            violations.append(ConstraintViolation(
                "Protocol headers preserved", header_diff, 1e-6, header_diff < 1e-6
            ))

        return violations

    def all_satisfied(self, x_orig: np.ndarray, x_adv: np.ndarray) -> bool:
        """Return True if all constraints are satisfied."""
        return all(v.satisfied for v in self.validate(x_orig, x_adv))

    def compute_mse(self, x_orig: np.ndarray, x_adv: np.ndarray) -> float:
        """Compute MSE distortion between original and perturbed features."""
        return float(np.mean((x_orig - x_adv) ** 2))
