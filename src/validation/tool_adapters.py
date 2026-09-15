"""
APE-BAIT Security Tool Adapters
Adapter pattern implementations for 15+ security tools.
Each adapter provides a uniform interface:
  analyze(pcap_path) → List[Detection]

Supported tools:
  - Snort           (rule-based IDS)
  - Suricata        (rule-based IDS)
  - YARA            (pattern matching)
  - ClamAV          (signature AV)
  - Zeek/Bro        (network monitor)
  - ML IDS CNN      (sklearn/PyTorch)
  - ML IDS RF       (Random Forest)
  - ML IDS LSTM     (LSTM classifier)
  - Anomaly IForest (Isolation Forest)
  - Anomaly LOF     (Local Outlier Factor)
  - Anomaly AE      (Autoencoder)
  - MalConv         (deep malware detector)
  - EMBER LightGBM  (ML malware)
  - DGA Detector    (domain generation algorithm)
  - Port Scan Det.  (heuristic scan detector)
"""

from __future__ import annotations

import abc
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logger import get_logger
from src.validation.metrics import Detection


logger = get_logger("ape-bait.adapters")


# ─── Base Adapter ───────────────────────────────────────────────────────────

class BaseToolAdapter(abc.ABC):
    """Abstract base for all security tool adapters."""

    name: str = "unknown"

    @abc.abstractmethod
    def analyze(self, pcap_path: str) -> List[Detection]:
        """
        Run the security tool on a PCAP file and return detections.

        Args:
            pcap_path: Absolute path to PCAP file.

        Returns:
            List of Detection objects. Empty list = no alerts.
        """
        ...

    def is_available(self) -> bool:
        """Return True if the tool is installed and available."""
        return True

    def _run_subprocess(
        self,
        cmd: List[str],
        timeout: int = 60,
    ) -> Tuple[int, str, str]:
        """Run a subprocess and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.warning(f"Tool {self.name} timed out after {timeout}s")
            return -1, "", "TIMEOUT"
        except FileNotFoundError:
            logger.warning(f"Tool {self.name} binary not found: {cmd[0]}")
            return -1, "", "NOT_FOUND"
        except Exception as e:
            logger.error(f"Tool {self.name} subprocess error: {e}")
            return -1, "", str(e)


# ─── Snort Adapter ───────────────────────────────────────────────────────────

class SnortAdapter(BaseToolAdapter):
    """Adapter for Snort IDS."""

    name = "snort"

    def __init__(self, config_file: str = "/etc/snort/snort.conf", binary: str = "snort"):
        self.config_file = config_file
        self.binary = binary

    def is_available(self) -> bool:
        rc, _, _ = self._run_subprocess([self.binary, "--version"], timeout=5)
        return rc == 0

    def analyze(self, pcap_path: str) -> List[Detection]:
        cmd = [
            self.binary,
            "-q",
            "-A", "fast",
            "-c", self.config_file,
            "-r", pcap_path,
            "-l", "/tmp/snort_ape_bait/",
        ]
        os.makedirs("/tmp/snort_ape_bait", exist_ok=True)
        rc, stdout, stderr = self._run_subprocess(cmd, timeout=120)

        detections = []
        alert_file = "/tmp/snort_ape_bait/alert"
        if os.path.exists(alert_file):
            with open(alert_file) as f:
                for line in f:
                    if "[**]" in line:
                        detections.append(Detection(
                            tool=self.name,
                            alert_type=self._parse_snort_alert(line),
                            message=line.strip(),
                        ))
            os.remove(alert_file)
        return detections

    def _parse_snort_alert(self, line: str) -> str:
        match = re.search(r'\[\*\*\]\s*\[.*?\]\s*(.*?)\s*\[\*\*\]', line)
        return match.group(1).strip() if match else "ALERT"


# ─── Suricata Adapter ────────────────────────────────────────────────────────

class SuricataAdapter(BaseToolAdapter):
    """Adapter for Suricata IDS."""

    name = "suricata"

    def __init__(
        self,
        config_file: str = "/etc/suricata/suricata.yaml",
        binary: str = "suricata",
        log_dir: str = "/tmp/suricata_ape_bait/",
    ):
        self.config_file = config_file
        self.binary = binary
        self.log_dir = log_dir

    def is_available(self) -> bool:
        rc, _, _ = self._run_subprocess([self.binary, "--version"], timeout=5)
        return rc == 0

    def analyze(self, pcap_path: str) -> List[Detection]:
        os.makedirs(self.log_dir, exist_ok=True)
        cmd = [
            self.binary,
            "-r", pcap_path,
            "-c", self.config_file,
            "-l", self.log_dir,
            "--runmode", "single",
        ]
        self._run_subprocess(cmd, timeout=120)

        detections = []
        eve_path = os.path.join(self.log_dir, "eve.json")
        if os.path.exists(eve_path):
            with open(eve_path) as f:
                for line in f:
                    try:
                        evt = json.loads(line)
                        if evt.get("event_type") == "alert":
                            alert = evt.get("alert", {})
                            detections.append(Detection(
                                tool=self.name,
                                alert_type=alert.get("signature", "ALERT"),
                                severity=alert.get("severity", 1),
                                src_ip=evt.get("src_ip", ""),
                                dst_ip=evt.get("dest_ip", ""),
                                protocol=evt.get("proto", ""),
                                message=alert.get("signature", ""),
                            ))
                    except json.JSONDecodeError:
                        continue
            os.remove(eve_path)
        return detections


# ─── YARA Adapter ────────────────────────────────────────────────────────────

class YARAAdapter(BaseToolAdapter):
    """Adapter for YARA rule-based pattern matching."""

    name = "yara"

    def __init__(self, rules_dir: str = "/etc/yara/rules/", binary: str = "yara"):
        self.rules_dir = rules_dir
        self.binary = binary

    def is_available(self) -> bool:
        rc, _, _ = self._run_subprocess([self.binary, "--version"], timeout=5)
        return rc == 0

    def analyze(self, pcap_path: str) -> List[Detection]:
        """Run YARA rules against extracted payloads from PCAP."""
        try:
            import yara  # type: ignore
        except ImportError:
            return self._analyze_subprocess(pcap_path)

        return self._analyze_yara_python(pcap_path)

    def _analyze_yara_python(self, pcap_path: str) -> List[Detection]:
        """Use yara-python for analysis."""
        import yara
        rules_path = Path(self.rules_dir)
        if not rules_path.exists():
            return []

        # Compile all .yar/.yara files
        filepaths = {}
        for yar_file in rules_path.glob("**/*.yar"):
            filepaths[yar_file.stem] = str(yar_file)
        for yar_file in rules_path.glob("**/*.yara"):
            filepaths[yar_file.stem] = str(yar_file)

        if not filepaths:
            return []

        try:
            rules = yara.compile(filepaths=filepaths)
        except Exception as e:
            logger.warning(f"YARA compile error: {e}")
            return []

        # Extract payload bytes and scan
        detections = []
        payloads = self._extract_payloads(pcap_path)
        for payload in payloads:
            matches = rules.match(data=payload)
            for match in matches:
                detections.append(Detection(
                    tool=self.name,
                    alert_type=match.rule,
                    message=f"YARA rule matched: {match.rule}",
                ))
        return detections

    def _analyze_subprocess(self, pcap_path: str) -> List[Detection]:
        """Fallback: run yara binary."""
        rules = list(Path(self.rules_dir).glob("**/*.yar")) if Path(self.rules_dir).exists() else []
        if not rules:
            return []

        detections = []
        cmd = [self.binary] + [str(r) for r in rules] + [pcap_path]
        rc, stdout, _ = self._run_subprocess(cmd, timeout=60)
        for line in stdout.splitlines():
            if line.strip():
                detections.append(Detection(
                    tool=self.name,
                    alert_type="YARA_MATCH",
                    message=line.strip(),
                ))
        return detections

    def _extract_payloads(self, pcap_path: str) -> List[bytes]:
        """Extract TCP/UDP payload bytes from PCAP."""
        payloads = []
        try:
            from scapy.all import rdpcap, TCP, UDP, Raw
            pkts = rdpcap(pcap_path)
            for pkt in pkts:
                if pkt.haslayer(Raw):
                    payloads.append(bytes(pkt[Raw]))
        except Exception:
            pass
        return payloads


# ─── ClamAV Adapter ──────────────────────────────────────────────────────────

class ClamAVAdapter(BaseToolAdapter):
    """Adapter for ClamAV antivirus (payload extraction + scanning)."""

    name = "clamav"

    def __init__(self, binary: str = "clamscan", extract_dir: str = "/tmp/clamav_ape_bait/"):
        self.binary = binary
        self.extract_dir = extract_dir

    def is_available(self) -> bool:
        rc, _, _ = self._run_subprocess([self.binary, "--version"], timeout=5)
        return rc == 0

    def analyze(self, pcap_path: str) -> List[Detection]:
        os.makedirs(self.extract_dir, exist_ok=True)
        payloads = self._extract_payloads(pcap_path)

        detections = []
        for i, payload in enumerate(payloads):
            if not payload:
                continue
            tmp_file = os.path.join(self.extract_dir, f"payload_{i}.bin")
            with open(tmp_file, "wb") as f:
                f.write(payload)

            rc, stdout, _ = self._run_subprocess([self.binary, tmp_file], timeout=30)
            if "FOUND" in stdout:
                for line in stdout.splitlines():
                    if "FOUND" in line:
                        detections.append(Detection(
                            tool=self.name,
                            alert_type="CLAMAV_DETECT",
                            message=line.strip(),
                        ))
            os.remove(tmp_file)

        return detections

    def _extract_payloads(self, pcap_path: str) -> List[bytes]:
        payloads = []
        try:
            from scapy.all import rdpcap, Raw
            pkts = rdpcap(pcap_path)
            for pkt in pkts:
                if pkt.haslayer(Raw):
                    payloads.append(bytes(pkt[Raw]))
        except Exception:
            pass
        return payloads


# ─── Zeek Adapter ────────────────────────────────────────────────────────────

class ZeekAdapter(BaseToolAdapter):
    """Adapter for Zeek (formerly Bro) network analysis framework."""

    name = "zeek"

    def __init__(self, binary: str = "zeek", log_dir: str = "/tmp/zeek_ape_bait/"):
        self.binary = binary
        self.log_dir = log_dir

    def is_available(self) -> bool:
        rc, _, _ = self._run_subprocess([self.binary, "--version"], timeout=5)
        return rc == 0

    def analyze(self, pcap_path: str) -> List[Detection]:
        os.makedirs(self.log_dir, exist_ok=True)
        cmd = [self.binary, "-r", pcap_path]
        rc, stdout, stderr = self._run_subprocess(cmd, timeout=120)

        detections = []
        notice_log = os.path.join(self.log_dir, "notice.log")
        weird_log = os.path.join(self.log_dir, "weird.log")

        for log_file in [notice_log, weird_log]:
            if os.path.exists(log_file):
                with open(log_file) as f:
                    for line in f:
                        if not line.startswith("#"):
                            detections.append(Detection(
                                tool=self.name,
                                alert_type="ZEEK_NOTICE",
                                message=line.strip(),
                            ))
        return detections


# ─── ML-Based Adapters ───────────────────────────────────────────────────────

class MLIDSCNNAdapter(BaseToolAdapter):
    """
    ML-based IDS using CNN surrogate model.
    Used to measure evasion against our own surrogate (sanity check).
    """

    name = "ml_ids_cnn"

    def __init__(self, model_path: Optional[str] = None, feature_dim: int = 256):
        self.model_path = model_path
        self.feature_dim = feature_dim
        self._model = None

    def _load_model(self):
        from src.models.surrogate_ids import build_surrogate_ids
        self._model = build_surrogate_ids(
            input_dim=self.feature_dim,
            pretrained_path=self.model_path,
        )
        self._model.eval()

    def analyze(self, pcap_path: str) -> List[Detection]:
        import torch
        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor

        if self._model is None:
            self._load_model()

        sniffer = PacketSniffer()
        preprocessor = PacketPreprocessor(feature_dim=self.feature_dim)

        try:
            packets = sniffer.read_pcap(pcap_path)
        except Exception:
            return []

        detections = []
        for pkt in packets:
            feat = preprocessor.extract(pkt)
            if feat is None:
                continue
            with torch.no_grad():
                x = torch.from_numpy(feat).unsqueeze(0)
                pred = self._model.predict(x).item()
                if pred == 1:  # Malicious
                    detections.append(Detection(
                        tool=self.name,
                        alert_type="CNN_MALICIOUS",
                        message="CNN IDS classified packet as malicious",
                    ))
        return detections


class MLIDSRandomForestAdapter(BaseToolAdapter):
    """ML-based IDS using Random Forest (sklearn)."""

    name = "ml_ids_rf"

    def __init__(self, model_path: Optional[str] = None, feature_dim: int = 256):
        self.model_path = model_path
        self.feature_dim = feature_dim
        self._model = None

    def _load_or_train_model(self):
        from sklearn.ensemble import RandomForestClassifier
        import pickle
        if self.model_path and Path(self.model_path).exists():
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
        else:
            # Train with synthetic data for demo
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=self.feature_dim)
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._model.fit(X, y)

    def analyze(self, pcap_path: str) -> List[Detection]:
        if self._model is None:
            self._load_or_train_model()

        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor

        sniffer = PacketSniffer()
        preprocessor = PacketPreprocessor(feature_dim=self.feature_dim)

        try:
            packets = sniffer.read_pcap(pcap_path)
        except Exception:
            return []

        detections = []
        features = []
        for pkt in packets:
            feat = preprocessor.extract(pkt)
            if feat is not None:
                features.append(feat)

        if not features:
            return []

        X = np.stack(features)
        preds = self._model.predict(X)
        for i, pred in enumerate(preds):
            if pred == 1:
                detections.append(Detection(
                    tool=self.name,
                    alert_type="RF_MALICIOUS",
                    message=f"Random Forest IDS flagged packet {i}",
                ))
        return detections


class AnomalyIsolationForestAdapter(BaseToolAdapter):
    """Anomaly detection using Isolation Forest."""

    name = "anomaly_iforest"

    def __init__(self, contamination: float = 0.1, feature_dim: int = 256):
        self.contamination = contamination
        self.feature_dim = feature_dim
        self._model = None

    def _train_model(self, normal_features: np.ndarray):
        from sklearn.ensemble import IsolationForest
        self._model = IsolationForest(contamination=self.contamination, random_state=42)
        self._model.fit(normal_features)

    def analyze(self, pcap_path: str) -> List[Detection]:
        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor

        sniffer = PacketSniffer()
        preprocessor = PacketPreprocessor(feature_dim=self.feature_dim)

        try:
            packets = sniffer.read_pcap(pcap_path)
        except Exception:
            return []

        features = []
        for pkt in packets:
            feat = preprocessor.extract(pkt)
            if feat is not None:
                features.append(feat)

        if not features:
            return []

        X = np.stack(features)

        if self._model is None:
            # Train on this traffic as "normal" baseline (demo mode)
            self._train_model(X)

        preds = self._model.predict(X)
        detections = []
        for i, pred in enumerate(preds):
            if pred == -1:  # Anomaly
                detections.append(Detection(
                    tool=self.name,
                    alert_type="IFOREST_ANOMALY",
                    message=f"Isolation Forest anomaly at packet {i}",
                ))
        return detections


class AnomalyLOFAdapter(BaseToolAdapter):
    """Anomaly detection using Local Outlier Factor."""

    name = "anomaly_lof"

    def __init__(self, n_neighbors: int = 20, feature_dim: int = 256):
        self.n_neighbors = n_neighbors
        self.feature_dim = feature_dim

    def analyze(self, pcap_path: str) -> List[Detection]:
        from sklearn.neighbors import LocalOutlierFactor
        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor

        sniffer = PacketSniffer()
        preprocessor = PacketPreprocessor(feature_dim=self.feature_dim)

        try:
            packets = sniffer.read_pcap(pcap_path)
        except Exception:
            return []

        features = []
        for pkt in packets:
            feat = preprocessor.extract(pkt)
            if feat is not None:
                features.append(feat)

        if len(features) < self.n_neighbors + 1:
            return []

        X = np.stack(features)
        clf = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=False)
        preds = clf.fit_predict(X)

        detections = []
        for i, pred in enumerate(preds):
            if pred == -1:
                detections.append(Detection(
                    tool=self.name,
                    alert_type="LOF_ANOMALY",
                    message=f"LOF anomaly at packet {i}",
                ))
        return detections


class AnomalyAutoencoderAdapter(BaseToolAdapter):
    """Anomaly detection using our SurrogateAnomalyDetector."""

    name = "anomaly_autoencoder"

    def __init__(self, threshold: float = 0.05, feature_dim: int = 256):
        self.threshold = threshold
        self.feature_dim = feature_dim
        self._model = None

    def _load_model(self):
        from src.models.surrogate_anomaly import build_surrogate_anomaly
        self._model = build_surrogate_anomaly(
            input_dim=self.feature_dim,
            threshold=self.threshold,
        )
        self._model.eval()

    def analyze(self, pcap_path: str) -> List[Detection]:
        import torch
        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor

        if self._model is None:
            self._load_model()

        sniffer = PacketSniffer()
        preprocessor = PacketPreprocessor(feature_dim=self.feature_dim)

        try:
            packets = sniffer.read_pcap(pcap_path)
        except Exception:
            return []

        detections = []
        for pkt in packets:
            feat = preprocessor.extract(pkt)
            if feat is None:
                continue
            with torch.no_grad():
                x = torch.from_numpy(feat).unsqueeze(0)
                is_anom = self._model.is_anomaly(x).item()
                if is_anom:
                    detections.append(Detection(
                        tool=self.name,
                        alert_type="AE_ANOMALY",
                        message="Autoencoder anomaly detected",
                    ))
        return detections


class MalConvAdapter(BaseToolAdapter):
    """Malware detection using MalConv-inspired surrogate."""

    name = "malconv"

    def __init__(self, model_path: Optional[str] = None, feature_dim: int = 256):
        self.model_path = model_path
        self.feature_dim = feature_dim
        self._model = None

    def _load_model(self):
        from src.models.surrogate_malware import build_surrogate_malware
        self._model = build_surrogate_malware(
            input_dim=self.feature_dim,
            pretrained_path=self.model_path,
        )
        self._model.eval()

    def analyze(self, pcap_path: str) -> List[Detection]:
        import torch
        from src.capture.sniffer import PacketSniffer
        from src.capture.preprocessor import PacketPreprocessor

        if self._model is None:
            self._load_model()

        sniffer = PacketSniffer()
        preprocessor = PacketPreprocessor(feature_dim=self.feature_dim)

        try:
            packets = sniffer.read_pcap(pcap_path)
        except Exception:
            return []

        detections = []
        for pkt in packets:
            feat = preprocessor.extract(pkt)
            if feat is None:
                continue
            with torch.no_grad():
                x = torch.from_numpy(feat).unsqueeze(0)
                pred = self._model(x).argmax(dim=-1).item()
                if pred != 0:  # Not benign
                    detections.append(Detection(
                        tool=self.name,
                        alert_type="MALCONV_DETECT",
                        message=f"MalConv detected malware class {pred}",
                    ))
        return detections


class DGADetectorAdapter(BaseToolAdapter):
    """Domain Generation Algorithm (DGA) detector based on DNS query entropy."""

    name = "dga_detector"

    def __init__(self, entropy_threshold: float = 3.5):
        self.entropy_threshold = entropy_threshold

    def analyze(self, pcap_path: str) -> List[Detection]:
        detections = []
        try:
            from scapy.all import rdpcap, DNS, DNSQR
            pkts = rdpcap(pcap_path)
            for pkt in pkts:
                if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
                    domain = pkt[DNSQR].qname.decode(errors='ignore').rstrip('.')
                    entropy = self._string_entropy(domain)
                    if entropy > self.entropy_threshold:
                        detections.append(Detection(
                            tool=self.name,
                            alert_type="DGA_SUSPECT",
                            message=f"High entropy domain: {domain} (H={entropy:.2f})",
                        ))
        except Exception:
            pass
        return detections

    def _string_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        counts = {}
        for c in s.lower():
            counts[c] = counts.get(c, 0) + 1
        total = len(s)
        return -sum((c / total) * np.log2(c / total) for c in counts.values())


class PortScanDetectorAdapter(BaseToolAdapter):
    """Heuristic port scan detector based on connection diversity."""

    name = "port_scan_detector"

    def __init__(self, unique_port_threshold: int = 15):
        self.unique_port_threshold = unique_port_threshold

    def analyze(self, pcap_path: str) -> List[Detection]:
        detections = []
        try:
            from scapy.all import rdpcap, TCP, IP
            pkts = rdpcap(pcap_path)

            src_ports: Dict[str, set] = {}
            for pkt in pkts:
                if pkt.haslayer(IP) and pkt.haslayer(TCP):
                    src = pkt[IP].src
                    dst_port = pkt[TCP].dport
                    src_ports.setdefault(src, set()).add(dst_port)

            for src_ip, ports in src_ports.items():
                if len(ports) > self.unique_port_threshold:
                    detections.append(Detection(
                        tool=self.name,
                        alert_type="PORT_SCAN",
                        src_ip=src_ip,
                        message=f"Port scan detected from {src_ip}: {len(ports)} unique ports",
                    ))
        except Exception:
            pass
        return detections


# ─── Adapter Registry ────────────────────────────────────────────────────────

ADAPTER_REGISTRY: Dict[str, type] = {
    "snort": SnortAdapter,
    "suricata": SuricataAdapter,
    "yara": YARAAdapter,
    "clamav": ClamAVAdapter,
    "zeek": ZeekAdapter,
    "ml_ids_cnn": MLIDSCNNAdapter,
    "ml_ids_rf": MLIDSRandomForestAdapter,
    "ml_ids_lstm": MLIDSCNNAdapter,       # Placeholder (uses CNN)
    "anomaly_iforest": AnomalyIsolationForestAdapter,
    "anomaly_lof": AnomalyLOFAdapter,
    "anomaly_autoencoder": AnomalyAutoencoderAdapter,
    "malconv": MalConvAdapter,
    "ember_lgbm": MLIDSRandomForestAdapter,  # Placeholder (uses RF)
    "dga_detector": DGADetectorAdapter,
    "port_scan_detector": PortScanDetectorAdapter,
}


def get_adapter(name: str, **kwargs) -> BaseToolAdapter:
    """Get an adapter by name from the registry."""
    cls = ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown tool adapter: {name}. Available: {list(ADAPTER_REGISTRY.keys())}")
    return cls(**kwargs)
