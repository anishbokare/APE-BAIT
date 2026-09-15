"""Pydantic schemas for APE-BAIT Interactive API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class AttackMethod(str, Enum):
    FGSM = "fgsm"
    PGD = "pgd"
    ENSEMBLE = "ensemble"


class TrafficSource(str, Enum):
    UPLOAD = "upload"
    LIVE = "live"
    SAMPLE = "sample"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class TargetTool(str, Enum):
    SNORT = "snort"
    SURICATA = "suricata"
    YARA = "yara"
    CLAMAV = "clamav"
    ZEEK = "zeek"
    MALCONV = "malconv"
    CNN_IDS = "cnn_ids"
    RANDOM_FOREST = "random_forest"
    LSTM_IDS = "lstm_ids"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    AUTOENCODER = "autoencoder"
    EMBER_GBM = "ember_gbm"
    DGA_DETECTOR = "dga_detector"
    PORT_SCAN_DETECTOR = "port_scan_detector"


# ── Request Schemas ────────────────────────────────────────────────────────────

class AttackParams(BaseModel):
    """Adversarial attack tuning parameters."""
    method: AttackMethod = Field(AttackMethod.PGD, description="Attack algorithm")
    epsilon: float = Field(0.03, ge=0.001, le=0.1, description="Perturbation budget (L-inf norm)")
    alpha: float = Field(0.005, ge=0.001, le=0.05, description="Step size per iteration (PGD only)")
    iterations: int = Field(10, ge=1, le=100, description="Number of PGD iterations")
    random_start: bool = Field(True, description="Use random initialization (PGD only)")
    target_tools: List[TargetTool] = Field(
        default=list(TargetTool),
        description="Security tools to evaluate evasion against",
    )

    @validator("alpha")
    def alpha_lt_epsilon(cls, v, values):
        if "epsilon" in values and v >= values["epsilon"]:
            raise ValueError("alpha must be less than epsilon")
        return v


class CreateJobRequest(BaseModel):
    """Request body for creating a new perturbation job."""
    traffic_source: TrafficSource = Field(TrafficSource.UPLOAD)
    live_interface: Optional[str] = Field(None, description="Network interface for live capture (e.g., eth0)")
    capture_duration_s: int = Field(30, ge=5, le=300, description="Live capture duration in seconds")
    attack_params: AttackParams = Field(default_factory=AttackParams)
    description: Optional[str] = Field(None, max_length=500)


# ── Response Schemas ───────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """Per-tool evasion result."""
    tool: TargetTool
    original_detections: int
    perturbed_detections: int
    evasion_rate: float
    execution_time_s: float
    error: Optional[str] = None


class DistortionStats(BaseModel):
    mean_mse: float
    p95_mse: float
    max_mse: float
    below_threshold: bool


class LatencyStats(BaseModel):
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    below_budget: bool


class JobSummary(BaseModel):
    """Summary metrics for a completed job."""
    aggregate_evasion_rate: float
    tools_tested: int
    tools_evaded: int
    distortion: DistortionStats
    latency: LatencyStats


class JobResponse(BaseModel):
    """Full job status and results response."""
    job_id: UUID
    status: JobStatus
    created_at: str
    updated_at: str
    pcap_filename: Optional[str] = None
    attack_params: AttackParams
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    progress_message: Optional[str] = None
    summary: Optional[JobSummary] = None
    tool_results: Optional[List[ToolResult]] = None
    perturbed_pcap_url: Optional[str] = None
    report_url: Optional[str] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int


class ToolInfo(BaseModel):
    """Metadata about a supported target tool."""
    id: TargetTool
    name: str
    category: str
    description: str
    typical_evasion_rate: str
