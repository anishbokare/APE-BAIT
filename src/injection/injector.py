"""
APE-BAIT Traffic Injector
Applies adversarial perturbations to raw packets and re-emits them.

Responsibilities:
  - Map feature-space perturbation deltas back to packet payload bytes
  - Recalculate IP/TCP/UDP checksums for modified packets
  - Write perturbed packets to PCAP or re-emit on live interface
  - Rate limiting to stay within configured PPS budget
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from src.core.config import InjectionConfig
from src.core.logger import get_logger
from src.injection.stealth import StealthValidator

try:
    from scapy.all import Packet as ScapyPacket, sendp, wrpcap, IP, TCP, UDP, Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class TrafficInjector:
    """
    Applies adversarial feature perturbations back to raw packets
    and handles re-emission or PCAP writing.

    Features:
      - Payload byte modification from feature delta
      - IP/TCP/UDP checksum recalculation
      - Rate-limited live injection (PPS control)
      - PCAP write mode for offline analysis
    """

    def __init__(self, config: InjectionConfig):
        self.config = config
        self.logger = get_logger("ape-bait.injector")
        self.stealth = StealthValidator()
        self._last_inject_time = 0.0
        self._pps_interval = 1.0 / max(1, config.rate_limit_pps)

    # ─── Public API ─────────────────────────────────────────────

    def apply(
        self,
        packet: Any,
        perturbed_features: np.ndarray,
        original_features: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Apply feature perturbation to a packet.

        The feature vector encodes payload bytes in positions [8:136].
        We map the perturbed feature values back to byte values and
        modify the packet payload accordingly.

        Args:
            packet: Original Scapy packet or raw bytes.
            perturbed_features: Perturbed feature vector (feature_dim,).
            original_features: Original features (optional, for delta computation).

        Returns:
            Modified packet (same type as input).
        """
        raw = self._to_bytes(packet)
        if not raw:
            return packet

        # Find payload offset
        payload_offset = self._get_payload_offset(raw)
        raw_arr = bytearray(raw)

        # Apply perturbed payload feature values (positions [8:136]) to bytes
        payload_features = perturbed_features[8:136]  # 128-byte payload slice
        for i, feat_val in enumerate(payload_features):
            byte_idx = payload_offset + i
            if byte_idx >= len(raw_arr):
                break
            # Denormalize: feature [0,1] → byte [0,255]
            raw_arr[byte_idx] = int(np.clip(feat_val * 255.0, 0, 255))

        modified = bytes(raw_arr)

        # Recalculate checksums if configured
        if self.config.recalc_checksums:
            modified = self._recalc_checksums(modified)

        # Reconstruct as same type as input
        return self._reconstruct(modified, packet)

    def inject_live(self, packet: Any) -> bool:
        """
        Re-emit a modified packet on the live network interface.

        Applies rate limiting before sending.

        Args:
            packet: Modified Scapy packet or raw bytes.

        Returns:
            True if sent successfully.
        """
        if not SCAPY_AVAILABLE:
            self.logger.warning("Scapy unavailable — live injection disabled")
            return False

        # Rate limiting
        now = time.time()
        elapsed = now - self._last_inject_time
        if elapsed < self._pps_interval:
            time.sleep(self._pps_interval - elapsed)
        self._last_inject_time = time.time()

        try:
            if hasattr(packet, 'haslayer'):
                sendp(packet, verbose=False)
            else:
                # Raw bytes: wrap in Ether
                sendp(Ether(bytes(packet)), verbose=False)
            return True
        except Exception as e:
            self.logger.error("Live injection failed", error=str(e))
            return False

    def write_pcap(self, packets: List[Any], output_path: str) -> None:
        """
        Write a list of modified packets to a PCAP file.

        Args:
            packets: List of Scapy packets or raw bytes.
            output_path: Destination file path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if SCAPY_AVAILABLE:
            scapy_packets = []
            for pkt in packets:
                if hasattr(pkt, 'haslayer'):
                    scapy_packets.append(pkt)
                else:
                    try:
                        scapy_packets.append(Ether(bytes(pkt)))
                    except Exception:
                        pass
            if scapy_packets:
                wrpcap(output_path, scapy_packets)
                self.logger.info(f"Wrote {len(scapy_packets)} packets to {output_path}")
                return

        # Fallback: raw PCAP write
        self._write_raw_pcap(packets, output_path)
        self.logger.info(f"Wrote {len(packets)} packets (raw) to {output_path}")

    # ─── Checksum Operations ────────────────────────────────────

    def _recalc_checksums(self, raw: bytes) -> bytes:
        """
        Recalculate IP, TCP, and UDP checksums for a modified packet.
        Returns packet with corrected checksums.
        """
        try:
            if SCAPY_AVAILABLE:
                # Easiest: use Scapy's auto-checksum
                pkt = Ether(raw)
                # Delete computed checksums to force recalculation
                if pkt.haslayer(IP):
                    del pkt[IP].chksum
                    if pkt.haslayer(TCP):
                        del pkt[TCP].chksum
                    elif pkt.haslayer(UDP):
                        del pkt[UDP].chksum
                # Re-build triggers checksum computation
                return bytes(pkt)
            else:
                return self._manual_checksum_recalc(raw)
        except Exception as e:
            self.logger.debug(f"Checksum recalc failed: {e}")
            return raw

    def _manual_checksum_recalc(self, raw: bytes) -> bytes:
        """Manual IP checksum recalculation without Scapy."""
        raw_arr = bytearray(raw)

        if len(raw_arr) < 34:
            return raw

        ip_offset = 14
        ip_byte = raw_arr[ip_offset]
        ip_version = (ip_byte >> 4) & 0xF
        if ip_version != 4:
            return raw

        ip_ihl = (ip_byte & 0xF) * 4

        # Zero out existing IP checksum
        raw_arr[ip_offset + 10] = 0
        raw_arr[ip_offset + 11] = 0

        # Compute new IP checksum
        ip_header = bytes(raw_arr[ip_offset:ip_offset + ip_ihl])
        new_chk = self._checksum(ip_header)
        raw_arr[ip_offset + 10] = (new_chk >> 8) & 0xFF
        raw_arr[ip_offset + 11] = new_chk & 0xFF

        return bytes(raw_arr)

    @staticmethod
    def _checksum(data: bytes) -> int:
        """Compute Internet checksum (RFC 1071)."""
        if len(data) % 2 != 0:
            data += b'\x00'
        words = struct.unpack(f"!{len(data)//2}H", data)
        total = sum(words)
        total = (total >> 16) + (total & 0xFFFF)
        total += total >> 16
        return ~total & 0xFFFF

    # ─── Internal Helpers ────────────────────────────────────────

    def _to_bytes(self, packet: Any) -> Optional[bytes]:
        if isinstance(packet, (bytes, bytearray)):
            return bytes(packet)
        try:
            return bytes(packet)
        except Exception:
            return None

    def _reconstruct(self, modified: bytes, original: Any) -> Any:
        """Reconstruct packet in same format as original."""
        if isinstance(original, (bytes, bytearray)):
            return modified
        if SCAPY_AVAILABLE and hasattr(original, 'haslayer'):
            try:
                return Ether(modified)
            except Exception:
                pass
        return modified

    def _get_payload_offset(self, raw: bytes) -> int:
        """Find byte offset where application payload begins."""
        if len(raw) < 14:
            return len(raw)

        try:
            ethertype = struct.unpack("!H", raw[12:14])[0]
            ip_offset = 14 if ethertype != 0x8100 else 18

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
            return transport_offset
        except Exception:
            return 14

    def _write_raw_pcap(self, packets: List[Any], output_path: str) -> None:
        """Write raw bytes as minimal PCAP format."""
        with open(output_path, "wb") as f:
            # PCAP global header
            f.write(struct.pack("IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
            ts = int(time.time())
            for pkt in packets:
                raw = bytes(pkt) if not isinstance(pkt, bytes) else pkt
                f.write(struct.pack("IIII", ts, 0, len(raw), len(raw)))
                f.write(raw)
