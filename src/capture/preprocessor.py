"""
APE-BAIT Packet Preprocessor
Converts raw network packets to fixed-size feature tensors
suitable for adversarial ML computation.
"""

from __future__ import annotations

import struct
from typing import Any, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.logger import get_logger


# ─── Feature Schema ─────────────────────────────────────────────────────────
# Feature vector layout (256 dimensions):
#
#  [0]        packet_len_norm       (1)
#  [1]        ip_proto_norm         (1)
#  [2-5]      tcp_flags (SYN,ACK,FIN,RST,PSH,URG encoded) (6 → padded to 4 slots)
#  [6]        ttl_norm              (1)
#  [7]        window_size_norm      (1)
#  [8-135]    payload_bytes_norm    (128 bytes, zero-padded)
#  [136-255]  inter_arrival_features / reserved (120 dims, zero-padded)
#
# Total: 256 dimensions

FEATURE_DIM = 256
PAYLOAD_SLICE = 128  # First 128 payload bytes
TCP_FLAG_BITS = ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]


class PacketPreprocessor:
    """
    Transforms raw Scapy packets or byte buffers into normalized feature tensors.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        payload_max_len: int = 512,
        device: str = "cpu",
    ):
        self.feature_dim = feature_dim
        self.payload_max_len = payload_max_len
        self.device = device
        self.logger = get_logger("ape-bait.preprocessor")

    # ─── Public API ─────────────────────────────────────────────

    def extract(self, packet: Any) -> Optional[np.ndarray]:
        """
        Extract feature vector from a packet.

        Args:
            packet: Scapy Packet object or raw bytes.

        Returns:
            Feature array of shape (feature_dim,), dtype float32. None if extraction fails.
        """
        try:
            raw_bytes = self._to_bytes(packet)
            if raw_bytes is None or len(raw_bytes) < 14:  # Minimum Ethernet frame
                return None
            features = self._extract_features(raw_bytes)
            return features.astype(np.float32)
        except Exception as e:
            self.logger.debug(f"Feature extraction failed: {e}")
            return None

    def extract_batch(self, packets: list) -> np.ndarray:
        """
        Extract features for a batch of packets.

        Returns:
            Array of shape (N, feature_dim), float32.
        """
        features = []
        for pkt in packets:
            f = self.extract(pkt)
            if f is not None:
                features.append(f)
            else:
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
        return np.stack(features, axis=0)

    def to_tensor(self, features: np.ndarray) -> Any:
        """Convert numpy array to PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor conversion.")
        import torch
        return torch.from_numpy(features).to(self.device)

    def from_tensor(self, tensor: Any) -> np.ndarray:
        """Convert PyTorch tensor back to numpy."""
        if TORCH_AVAILABLE:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def apply_perturbation(self, packet: Any, original_features: np.ndarray, perturbed_features: np.ndarray) -> Any:
        """
        Apply the delta between original and perturbed features back to the raw packet.
        Modifies the packet's payload bytes according to the feature perturbation.

        Args:
            packet: Original Scapy packet or raw bytes.
            original_features: Original feature vector.
            perturbed_features: Adversarially perturbed feature vector.

        Returns:
            Modified packet (same type as input).
        """
        raw = self._to_bytes(packet)
        if raw is None:
            return packet

        # Extract payload delta from feature positions [8:8+PAYLOAD_SLICE]
        original_payload_feat = original_features[8:8 + PAYLOAD_SLICE]
        perturbed_payload_feat = perturbed_features[8:8 + PAYLOAD_SLICE]
        delta = perturbed_payload_feat - original_payload_feat

        # Find payload offset in raw bytes (skip Ethernet + IP headers)
        payload_offset = self._get_payload_offset(raw)
        raw_list = bytearray(raw)

        for i, d in enumerate(delta):
            byte_idx = payload_offset + i
            if byte_idx >= len(raw_list):
                break
            # Scale delta back to byte range and clip
            new_val = raw_list[byte_idx] + int(d * 255.0)
            raw_list[byte_idx] = max(0, min(255, new_val))

        modified = bytes(raw_list)

        # Try to reconstruct as Scapy packet
        try:
            from scapy.all import Ether
            return Ether(modified)
        except Exception:
            return modified

    # ─── Internal Feature Extraction ────────────────────────────

    def _to_bytes(self, packet: Any) -> Optional[bytes]:
        """Convert any packet representation to raw bytes."""
        if isinstance(packet, bytes):
            return packet
        if isinstance(packet, bytearray):
            return bytes(packet)
        try:
            # Scapy packet
            return bytes(packet)
        except Exception:
            return None

    def _extract_features(self, raw: bytes) -> np.ndarray:
        """Extract and normalize features from raw packet bytes."""
        feat = np.zeros(self.feature_dim, dtype=np.float64)

        # ── Ethernet Frame ──────────────────────────────────────
        if len(raw) < 14:
            return feat.astype(np.float32)

        ethertype = struct.unpack("!H", raw[12:14])[0]

        ip_offset = 14  # Standard Ethernet header
        if ethertype == 0x8100:  # VLAN tag
            ip_offset = 18

        # ── IP Header ───────────────────────────────────────────
        if len(raw) < ip_offset + 20:
            return feat.astype(np.float32)

        ip_byte = raw[ip_offset]
        ip_version = (ip_byte >> 4) & 0xF
        ip_ihl = (ip_byte & 0xF) * 4

        if ip_version == 4 and len(raw) >= ip_offset + 20:
            total_len = struct.unpack("!H", raw[ip_offset + 2:ip_offset + 4])[0]
            ttl = raw[ip_offset + 8]
            protocol = raw[ip_offset + 9]

            feat[0] = min(total_len / 65535.0, 1.0)   # packet length (norm)
            feat[1] = protocol / 255.0                  # IP protocol (norm)
            feat[6] = ttl / 255.0                       # TTL (norm)

            transport_offset = ip_offset + ip_ihl

            # ── TCP ─────────────────────────────────────────────
            if protocol == 6 and len(raw) >= transport_offset + 20:
                flags = raw[transport_offset + 13]
                feat[2] = (flags >> 1) & 1   # SYN
                feat[3] = (flags >> 4) & 1   # ACK
                feat[4] = (flags >> 0) & 1   # FIN
                feat[5] = (flags >> 2) & 1   # RST
                window = struct.unpack("!H", raw[transport_offset + 14:transport_offset + 16])[0]
                feat[7] = window / 65535.0
                tcp_doffset = ((raw[transport_offset + 12] >> 4) & 0xF) * 4
                payload_start = transport_offset + tcp_doffset
            # ── UDP ─────────────────────────────────────────────
            elif protocol == 17 and len(raw) >= transport_offset + 8:
                payload_start = transport_offset + 8
            else:
                payload_start = transport_offset

            # ── Payload Bytes (normalized) ───────────────────────
            payload = raw[payload_start:payload_start + PAYLOAD_SLICE]
            for i, b in enumerate(payload):
                feat[8 + i] = b / 255.0

        elif ip_version == 6 and len(raw) >= ip_offset + 40:
            # IPv6: simplified extraction
            next_header = raw[ip_offset + 6]
            feat[1] = next_header / 255.0
            feat[6] = 1.0  # IPv6 has no TTL in same position; use hop limit
            hop_limit = raw[ip_offset + 7]
            feat[6] = hop_limit / 255.0
            payload_start = ip_offset + 40
            payload = raw[payload_start:payload_start + PAYLOAD_SLICE]
            for i, b in enumerate(payload):
                feat[8 + i] = b / 255.0

        return feat.astype(np.float32)

    def _get_payload_offset(self, raw: bytes) -> int:
        """Estimate the byte offset where application payload begins."""
        if len(raw) < 14:
            return len(raw)

        ethertype = struct.unpack("!H", raw[12:14])[0]
        ip_offset = 14
        if ethertype == 0x8100:
            ip_offset = 18

        if len(raw) < ip_offset + 20:
            return ip_offset

        ip_byte = raw[ip_offset]
        ip_version = (ip_byte >> 4) & 0xF
        ip_ihl = (ip_byte & 0xF) * 4
        protocol = raw[ip_offset + 9] if ip_version == 4 else 0

        transport_offset = ip_offset + ip_ihl

        if protocol == 6 and len(raw) >= transport_offset + 20:
            tcp_doffset = ((raw[transport_offset + 12] >> 4) & 0xF) * 4
            return transport_offset + tcp_doffset
        elif protocol == 17 and len(raw) >= transport_offset + 8:
            return transport_offset + 8
        else:
            return transport_offset
