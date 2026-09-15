"""
APE-BAIT Packet Sniffer
Scapy-based packet capture with BPF filter support, PCAP replay,
and async queue feeding.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

# Suppress Scapy warnings on import
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

try:
    from scapy.all import (
        AsyncSniffer,
        Packet,
        PcapReader,
        PcapWriter,
        conf,
        sniff,
        wrpcap,
        rdpcap,
    )
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    Packet = object  # type: ignore

from src.core.logger import get_logger


class PacketSniffer:
    """
    Packet capture component using Scapy.

    Supports:
      - Live capture on a network interface
      - PCAP file replay (for demo/testing)
      - BPF filter expressions
      - Async operation with callback for queue feeding
    """

    def __init__(
        self,
        interface: str = "eth0",
        bpf_filter: str = "",
        snapshot_len: int = 65535,
        promisc: bool = True,
        timeout_ms: int = 100,
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.snapshot_len = snapshot_len
        self.promisc = promisc
        self.timeout_ms = timeout_ms
        self.logger = get_logger("ape-bait.sniffer")
        self._sniffer: Optional[AsyncSniffer] = None

        if not SCAPY_AVAILABLE:
            self.logger.warning(
                "Scapy not installed — live capture disabled. "
                "PCAP replay still works via dpkt fallback."
            )

    # ─── Live Capture ───────────────────────────────────────────

    def start(self, callback: Callable) -> None:
        """
        Start live packet capture. Calls `callback(packet)` for each captured packet.
        Blocks until stopped or KeyboardInterrupt.

        Args:
            callback: Function to call with each captured Scapy Packet.
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for live capture. Install it: pip install scapy")

        self.logger.info(
            "Starting live capture",
            interface=self.interface,
            bpf_filter=self.bpf_filter or "(all traffic)",
        )

        kwargs = {
            "iface": self.interface,
            "prn": callback,
            "store": False,
            "snaplen": self.snapshot_len,
            "filter": self.bpf_filter,
        }
        if self.promisc:
            conf.promisc = True

        try:
            sniff(**kwargs)
        except PermissionError:
            self.logger.error(
                "Permission denied for live capture. "
                "Run as root/Administrator or use PCAP mode."
            )
            raise

    def start_async(self, callback: Callable) -> None:
        """Start non-blocking async sniffer."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for live capture.")

        self._sniffer = AsyncSniffer(
            iface=self.interface,
            prn=callback,
            store=False,
            filter=self.bpf_filter,
        )
        self._sniffer.start()
        self.logger.info("Async sniffer started", interface=self.interface)

    def stop_async(self) -> None:
        """Stop the async sniffer."""
        if self._sniffer:
            self._sniffer.stop()
            self.logger.info("Async sniffer stopped")

    # ─── PCAP File Operations ───────────────────────────────────

    def read_pcap(self, pcap_path: str) -> List[Any]:
        """
        Read packets from a PCAP file.

        Args:
            pcap_path: Path to .pcap or .pcapng file.

        Returns:
            List of packets (Scapy Packet objects if available, else raw bytes).
        """
        path = Path(pcap_path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        self.logger.info("Reading PCAP file", path=str(path))

        if SCAPY_AVAILABLE:
            try:
                packets = rdpcap(str(path))
                self.logger.info(f"Loaded {len(packets)} packets via Scapy")
                return list(packets)
            except Exception as e:
                self.logger.warning(f"Scapy read failed: {e}, falling back to dpkt")

        # Fallback: dpkt
        return self._read_pcap_dpkt(str(path))

    def _read_pcap_dpkt(self, pcap_path: str) -> List[bytes]:
        """Fallback PCAP reader using dpkt."""
        try:
            import dpkt
            packets = []
            with open(pcap_path, "rb") as f:
                reader = dpkt.pcap.Reader(f)
                for ts, buf in reader:
                    packets.append(buf)
            self.logger.info(f"Loaded {len(packets)} packets via dpkt")
            return packets
        except ImportError:
            raise RuntimeError("Neither Scapy nor dpkt is available for PCAP reading.")

    def write_pcap(self, packets: List[Any], output_path: str) -> None:
        """
        Write packets to a PCAP file.

        Args:
            packets: List of Scapy Packet objects or raw bytes.
            output_path: Output file path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if SCAPY_AVAILABLE and packets and hasattr(packets[0], "haslayer"):
            wrpcap(output_path, packets)
        else:
            # Write raw bytes with a minimal PCAP header
            self._write_raw_pcap(packets, output_path)

        self.logger.info(f"Wrote {len(packets)} packets to {output_path}")

    def _write_raw_pcap(self, packets: List[bytes], output_path: str) -> None:
        """Write raw byte packets as PCAP (dpkt or manual)."""
        try:
            import dpkt
            with open(output_path, "wb") as f:
                writer = dpkt.pcap.Writer(f)
                for pkt in packets:
                    writer.writepkt(pkt)
        except ImportError:
            # Manual minimal PCAP format
            import struct
            with open(output_path, "wb") as f:
                # Global header
                f.write(struct.pack("IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
                ts = int(time.time())
                for pkt in packets:
                    raw = bytes(pkt) if not isinstance(pkt, bytes) else pkt
                    f.write(struct.pack("IIII", ts, 0, len(raw), len(raw)))
                    f.write(raw)


# Type alias for external use
from typing import Any
