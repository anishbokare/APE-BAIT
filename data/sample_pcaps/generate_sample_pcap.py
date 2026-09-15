#!/usr/bin/env python
"""
Generate a minimal sample PCAP file for APE-BAIT demo/testing.
Creates synthetic TCP traffic that can be used to test the engine.

Run: python data/sample_pcaps/generate_sample_pcap.py
"""

import struct
import time
import random
import sys
import os
from pathlib import Path


def make_tcp_packet(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    payload: bytes = b"",
    flags: int = 0x002,  # SYN
    ttl: int = 64,
) -> bytes:
    """Build a minimal Ethernet + IP + TCP packet."""

    def ip_to_bytes(ip: str) -> bytes:
        return bytes(int(p) for p in ip.split("."))

    def checksum(data: bytes) -> int:
        if len(data) % 2 != 0:
            data += b'\x00'
        words = struct.unpack(f"!{len(data)//2}H", data)
        total = sum(words)
        total = (total >> 16) + (total & 0xFFFF)
        return ~total & 0xFFFF

    # Ethernet header
    eth = b'\xff\xff\xff\xff\xff\xff' + b'\x00\x11\x22\x33\x44\x55' + b'\x08\x00'

    # TCP header (no options, data offset = 5)
    tcp_seq = random.randint(0, 2**32 - 1)
    tcp_ack = 0
    tcp_doffset_flags = (5 << 12) | (flags & 0x3F)
    tcp_header_no_chk = struct.pack("!HHIIHH",
        src_port, dst_port, tcp_seq, tcp_ack,
        tcp_doffset_flags, 65535
    ) + b'\x00\x00\x00\x00'  # checksum + urgent

    # IP header
    ip_total_len = 20 + 20 + len(payload)
    ip_src = ip_to_bytes(src_ip)
    ip_dst = ip_to_bytes(dst_ip)

    ip_header_no_chk = struct.pack("!BBHHHBB",
        0x45, 0, ip_total_len,
        random.randint(0, 0xFFFF), 0x4000,
        ttl, 6
    ) + b'\x00\x00' + ip_src + ip_dst

    ip_chk = checksum(ip_header_no_chk)
    ip_header = ip_header_no_chk[:10] + struct.pack("!H", ip_chk) + ip_header_no_chk[12:]

    return eth + ip_header + tcp_header_no_chk + payload


def write_pcap(packets: list, path: str) -> None:
    """Write packets to PCAP file."""
    with open(path, "wb") as f:
        # PCAP global header
        f.write(struct.pack("IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        ts = int(time.time())
        for pkt in packets:
            f.write(struct.pack("IIII", ts, 0, len(pkt), len(pkt)))
            f.write(pkt)


def generate_sample_traffic(n_packets: int = 200) -> list:
    """Generate mixed benign and suspicious synthetic traffic."""
    packets = []
    rng = random.Random(42)

    # Benign HTTP-like traffic
    for i in range(n_packets // 2):
        payload = f"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n".encode()
        pkt = make_tcp_packet(
            src_ip=f"192.168.1.{rng.randint(1, 50)}",
            dst_ip="93.184.216.34",
            src_port=rng.randint(50000, 65000),
            dst_port=80,
            payload=payload,
            flags=0x018,  # PSH+ACK
        )
        packets.append(pkt)

    # Suspicious port scan-like traffic
    for i in range(n_packets // 4):
        pkt = make_tcp_packet(
            src_ip="10.0.0.1",
            dst_ip=f"192.168.1.{rng.randint(1, 254)}",
            src_port=rng.randint(50000, 65000),
            dst_port=rng.randint(1, 1024),
            payload=b"",
            flags=0x002,  # SYN
        )
        packets.append(pkt)

    # Suspicious high-entropy payload (simulated exploit attempt)
    for i in range(n_packets // 4):
        payload = bytes(rng.randint(0, 255) for _ in range(rng.randint(50, 200)))
        pkt = make_tcp_packet(
            src_ip="172.16.0.1",
            dst_ip="192.168.1.100",
            src_port=rng.randint(50000, 65000),
            dst_port=rng.randint(4444, 9999),
            payload=payload,
            flags=0x018,  # PSH+ACK
        )
        packets.append(pkt)

    rng.shuffle(packets)
    return packets


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_path = output_dir / "sample_traffic.pcap"

    print(f"Generating sample PCAP at {output_path}...")
    packets = generate_sample_traffic(n_packets=200)
    write_pcap(packets, str(output_path))
    print(f"[OK] Generated {len(packets)} packets -> {output_path}")
    print(f"   File size: {output_path.stat().st_size:,} bytes")
