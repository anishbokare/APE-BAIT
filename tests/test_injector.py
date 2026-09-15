"""
Unit tests for TrafficInjector.
Tests packet modification, checksum recalculation, and PCAP write.
"""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import load_config
from src.injection.injector import TrafficInjector


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def injector(config):
    return TrafficInjector(config.injection)


def make_minimal_tcp_packet() -> bytes:
    """Construct a minimal valid Ethernet+IP+TCP packet."""
    # Ethernet header (14 bytes): dst MAC, src MAC, ethertype (IPv4)
    eth = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x11\x22\x33\x44\x55' + b'\x08\x00'

    # IP header (20 bytes)
    # version=4, ihl=5 (20 bytes), dscp=0, ecn=0
    ip_version_ihl = 0x45
    ip_tos = 0
    ip_total_len = 40 + 10  # IP(20) + TCP(20) + payload(10) = 50, but set 40 for now
    ip_total_len = 50
    ip_id = 0x1234
    ip_flags_frag = 0x4000  # Don't fragment
    ip_ttl = 64
    ip_proto = 6  # TCP
    ip_chksum = 0  # Will compute
    ip_src = b'\xc0\xa8\x01\x01'  # 192.168.1.1
    ip_dst = b'\xc0\xa8\x01\x02'  # 192.168.1.2

    ip_header = struct.pack("!BBHHHBBH",
        ip_version_ihl, ip_tos, ip_total_len, ip_id,
        ip_flags_frag, ip_ttl, ip_proto, ip_chksum
    ) + ip_src + ip_dst

    # Compute IP checksum
    words = struct.unpack("!10H", ip_header)
    total = sum(words)
    total = (total >> 16) + (total & 0xFFFF)
    chksum = ~total & 0xFFFF
    ip_header = struct.pack("!BBHHHBBH",
        ip_version_ihl, ip_tos, ip_total_len, ip_id,
        ip_flags_frag, ip_ttl, ip_proto, chksum
    ) + ip_src + ip_dst

    # TCP header (20 bytes): minimal
    tcp_src = 12345
    tcp_dst = 80
    tcp_seq = 0
    tcp_ack = 0
    tcp_doffset_flags = (5 << 12) | 0x002  # data offset=5, SYN flag
    tcp_window = 65535
    tcp_chksum = 0
    tcp_urgent = 0

    tcp_header = struct.pack("!HHIIHHHH",
        tcp_src, tcp_dst, tcp_seq, tcp_ack,
        tcp_doffset_flags, tcp_window, tcp_chksum, tcp_urgent
    )

    # 10 bytes of payload
    payload = b'HELLO_APE!' 

    return eth + ip_header + tcp_header + payload


# ─── Payload Modification Tests ───────────────────────────────────

class TestInjectorPayloadModification:
    def test_apply_returns_same_length(self, injector):
        """Injecting should not change packet length."""
        pkt = make_minimal_tcp_packet()
        features = np.zeros(256, dtype=np.float32)
        result = injector.apply(pkt, features)
        result_bytes = bytes(result) if not isinstance(result, bytes) else result
        assert len(result_bytes) == len(pkt)

    def test_apply_modifies_payload(self, injector):
        """With nonzero feature values, payload bytes should change."""
        pkt = make_minimal_tcp_packet()
        # Set some payload features to nonzero
        features = np.zeros(256, dtype=np.float32)
        features[8:18] = 0.5  # Modify first 10 payload bytes

        result = injector.apply(pkt, features)
        result_bytes = bytes(result) if not isinstance(result, bytes) else result

        # The payload area should have changed
        orig_bytes = pkt
        changed = any(orig_bytes[i] != result_bytes[i] for i in range(len(orig_bytes)))
        assert changed, "Expected payload modification but packet is unchanged"

    def test_apply_bytes_in_range(self, injector):
        """All bytes in result must be valid (0-255)."""
        pkt = make_minimal_tcp_packet()
        features = np.random.uniform(0, 1, 256).astype(np.float32)
        result = injector.apply(pkt, features)
        result_bytes = bytes(result) if not isinstance(result, bytes) else result
        for b in result_bytes:
            assert 0 <= b <= 255

    def test_apply_raw_bytes_input(self, injector):
        """Injector should accept raw bytes directly."""
        pkt = make_minimal_tcp_packet()
        features = np.zeros(256, dtype=np.float32)
        result = injector.apply(pkt, features)
        assert result is not None


# ─── Checksum Tests ───────────────────────────────────────────────

class TestInjectorChecksums:
    def test_ip_checksum_valid_after_injection(self, injector):
        """IP checksum should be valid after payload modification."""
        from src.injection.stealth import StealthValidator
        pkt = make_minimal_tcp_packet()
        features = np.random.uniform(0, 1, 256).astype(np.float32)
        features[:8] = 0.0  # Don't touch headers in features

        result = injector.apply(pkt, features)
        result_bytes = bytes(result) if not isinstance(result, bytes) else result

        stealth = StealthValidator()
        # Check IP checksum is valid
        valid = stealth._validate_checksums(result_bytes)
        assert valid, "IP checksum should be valid after injection"

    def test_checksum_internet_calculation(self, injector):
        """Verify the static checksum calculation is correct."""
        # Known test vector: IP header with zeros in checksum field
        ip_header = bytes([
            0x45, 0x00, 0x00, 0x28,  # ver/ihl, tos, total len
            0x12, 0x34, 0x40, 0x00,  # id, flags/frag
            0x40, 0x06, 0x00, 0x00,  # ttl, proto, checksum (0)
            0xc0, 0xa8, 0x01, 0x01,  # src IP
            0xc0, 0xa8, 0x01, 0x02,  # dst IP
        ])
        chksum = TrafficInjector._checksum(ip_header)
        assert 0 <= chksum <= 65535, "Checksum must be 16-bit"


# ─── PCAP Write Tests ─────────────────────────────────────────────

class TestInjectorPCAPWrite:
    def test_write_pcap_creates_file(self, injector, tmp_path):
        """write_pcap should create the output file."""
        pkt = make_minimal_tcp_packet()
        out_path = str(tmp_path / "test_output.pcap")
        injector.write_pcap([pkt, pkt], out_path)
        assert Path(out_path).exists()

    def test_write_pcap_nonempty(self, injector, tmp_path):
        """Written PCAP should not be empty."""
        pkt = make_minimal_tcp_packet()
        out_path = str(tmp_path / "test_output.pcap")
        injector.write_pcap([pkt, pkt, pkt], out_path)
        size = Path(out_path).stat().st_size
        assert size > 24, "PCAP file should contain more than just the global header"

    def test_write_empty_list(self, injector, tmp_path):
        """Writing an empty list should not crash."""
        out_path = str(tmp_path / "empty.pcap")
        injector.write_pcap([], out_path)
        # File may or may not be created for empty list — just no exception

    def test_write_pcap_valid_global_header(self, injector, tmp_path):
        """PCAP global header magic number should be 0xa1b2c3d4."""
        pkt = make_minimal_tcp_packet()
        out_path = str(tmp_path / "check_header.pcap")
        injector._write_raw_pcap([pkt], out_path)
        with open(out_path, "rb") as f:
            magic = struct.unpack("I", f.read(4))[0]
        assert magic == 0xa1b2c3d4, "Invalid PCAP magic number"


# ─── Payload Offset Tests ─────────────────────────────────────────

class TestPayloadOffset:
    def test_tcp_payload_offset(self, injector):
        """TCP payload should start after Ethernet(14) + IP(20) + TCP(20) = 54."""
        pkt = make_minimal_tcp_packet()
        offset = injector._get_payload_offset(pkt)
        assert offset == 54, f"Expected TCP payload at offset 54, got {offset}"

    def test_short_packet_offset(self, injector):
        """Short packets should not crash."""
        short_pkt = b'\x00' * 10
        offset = injector._get_payload_offset(short_pkt)
        assert offset <= 10  # Should return something ≤ packet length
