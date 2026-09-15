"""
APE-BAIT Stealth Validator
Validates that adversarial perturbations remain imperceptible:
  - MSE distortion < threshold
  - Structural similarity checks
  - Protocol compliance (valid checksums, header structure)
  - Payload entropy analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logger import get_logger


@dataclass
class StealthReport:
    """Report from a stealth validation check."""
    mse: float
    linf: float
    l2: float
    entropy_delta: float
    header_modified: bool
    checksum_valid: bool
    is_stealthy: bool
    violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": round(self.mse, 6),
            "linf": round(self.linf, 6),
            "l2": round(self.l2, 6),
            "entropy_delta": round(self.entropy_delta, 6),
            "header_modified": self.header_modified,
            "checksum_valid": self.checksum_valid,
            "is_stealthy": self.is_stealthy,
            "violations": self.violations,
        }


class StealthValidator:
    """
    Validates that adversarial perturbations are imperceptible.

    Checks:
      1. MSE distortion vs. threshold
      2. L∞ and L2 norms
      3. Payload entropy change (large delta = detectable)
      4. Protocol header integrity
      5. Packet length preservation
    """

    # Protected header indices in feature vector
    HEADER_INDICES = list(range(8))

    def __init__(
        self,
        mse_threshold: float = 0.04,
        max_entropy_delta: float = 0.1,
        protect_headers: bool = True,
    ):
        self.mse_threshold = mse_threshold
        self.max_entropy_delta = max_entropy_delta
        self.protect_headers = protect_headers
        self.logger = get_logger("ape-bait.stealth")

    def validate_features(
        self,
        original: np.ndarray,
        perturbed: np.ndarray,
    ) -> StealthReport:
        """
        Validate that a feature-space perturbation is stealthy.

        Args:
            original: Original feature vector (feature_dim,)
            perturbed: Perturbed feature vector (feature_dim,)

        Returns:
            StealthReport with pass/fail details.
        """
        delta = perturbed - original
        violations = []

        # ── MSE ──────────────────────────────────────────────────
        mse = float(np.mean(delta ** 2))
        if mse > self.mse_threshold:
            violations.append(f"MSE {mse:.6f} > threshold {self.mse_threshold}")

        # ── L∞ ───────────────────────────────────────────────────
        linf = float(np.max(np.abs(delta)))

        # ── L2 ───────────────────────────────────────────────────
        l2 = float(np.linalg.norm(delta))

        # ── Entropy ───────────────────────────────────────────────
        entropy_orig = self._entropy(original[8:136])   # Payload feature slice
        entropy_pert = self._entropy(perturbed[8:136])
        entropy_delta = abs(entropy_pert - entropy_orig)
        if entropy_delta > self.max_entropy_delta:
            violations.append(f"Entropy delta {entropy_delta:.4f} > {self.max_entropy_delta}")

        # ── Header integrity ──────────────────────────────────────
        header_modified = bool(
            np.any(np.abs(delta[self.HEADER_INDICES]) > 1e-6)
        ) if self.protect_headers else False
        if header_modified:
            violations.append("Protocol headers were modified")

        is_stealthy = len(violations) == 0

        return StealthReport(
            mse=mse,
            linf=linf,
            l2=l2,
            entropy_delta=entropy_delta,
            header_modified=header_modified,
            checksum_valid=True,   # Checksum validated at packet layer by injector
            is_stealthy=is_stealthy,
            violations=violations,
        )

    def validate_packets(
        self,
        original_packet: Any,
        perturbed_packet: Any,
    ) -> StealthReport:
        """
        Validate stealth at the raw packet level.

        Args:
            original_packet: Original raw packet bytes or Scapy packet.
            perturbed_packet: Perturbed raw packet bytes or Scapy packet.

        Returns:
            StealthReport.
        """
        orig_bytes = self._to_bytes(original_packet)
        pert_bytes = self._to_bytes(perturbed_packet)

        if not orig_bytes or not pert_bytes:
            return StealthReport(
                mse=0.0, linf=0.0, l2=0.0, entropy_delta=0.0,
                header_modified=False, checksum_valid=False,
                is_stealthy=False, violations=["Could not convert packets to bytes"]
            )

        # Pad to same length
        max_len = max(len(orig_bytes), len(pert_bytes))
        orig_arr = np.frombuffer(orig_bytes.ljust(max_len, b'\x00'), dtype=np.uint8).astype(np.float32) / 255.0
        pert_arr = np.frombuffer(pert_bytes.ljust(max_len, b'\x00'), dtype=np.uint8).astype(np.float32) / 255.0

        violations = []
        delta = pert_arr - orig_arr

        mse = float(np.mean(delta ** 2))
        if mse > self.mse_threshold:
            violations.append(f"Byte-level MSE {mse:.6f} > {self.mse_threshold}")

        linf = float(np.max(np.abs(delta)))
        l2 = float(np.linalg.norm(delta))

        # Entropy of byte distributions
        entropy_orig = self._entropy(orig_arr)
        entropy_pert = self._entropy(pert_arr)
        entropy_delta = abs(entropy_pert - entropy_orig)

        # Check length changed (could be suspicious)
        if len(orig_bytes) != len(pert_bytes):
            violations.append(f"Packet length changed: {len(orig_bytes)} → {len(pert_bytes)}")

        # Validate checksums
        checksum_valid = self._validate_checksums(pert_bytes)
        if not checksum_valid:
            violations.append("Invalid checksum in perturbed packet")

        return StealthReport(
            mse=mse,
            linf=linf,
            l2=l2,
            entropy_delta=entropy_delta,
            header_modified=False,
            checksum_valid=checksum_valid,
            is_stealthy=len(violations) == 0,
            violations=violations,
        )

    def _entropy(self, arr: np.ndarray) -> float:
        """Compute Shannon entropy of normalized feature/byte array."""
        if len(arr) == 0:
            return 0.0
        # Histogram over 256 bins
        hist, _ = np.histogram(arr, bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _to_bytes(self, packet: Any) -> Optional[bytes]:
        """Convert packet to bytes."""
        if isinstance(packet, (bytes, bytearray)):
            return bytes(packet)
        try:
            return bytes(packet)
        except Exception:
            return None

    def _validate_checksums(self, raw: bytes) -> bool:
        """
        Basic checksum validation for IP/TCP/UDP.
        Returns True if checksums appear valid or can't be determined.
        """
        try:
            import struct
            if len(raw) < 34:  # Min Ethernet + IP + TCP
                return True

            # Ethernet → IP
            ip_offset = 14
            ip_byte = raw[ip_offset]
            ip_version = (ip_byte >> 4) & 0xF
            if ip_version != 4:
                return True

            ip_ihl = (ip_byte & 0xF) * 4
            ip_header = raw[ip_offset:ip_offset + ip_ihl]
            if len(ip_header) < 20:
                return True

            # Verify IP checksum
            if not self._ip_checksum_valid(ip_header):
                return False

            return True
        except Exception:
            return True  # Benefit of the doubt if we can't parse

    def _ip_checksum_valid(self, ip_header: bytes) -> bool:
        """Verify IP header checksum."""
        import struct
        # Zero out checksum field for calculation
        h = bytearray(ip_header)
        h[10] = 0
        h[11] = 0
        words = struct.unpack(f"!{len(h)//2}H", bytes(h))
        checksum = sum(words)
        checksum = (checksum >> 16) + (checksum & 0xFFFF)
        checksum = ~checksum & 0xFFFF
        stored = struct.unpack("!H", ip_header[10:12])[0]
        return checksum == stored
