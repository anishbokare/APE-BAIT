"""Jobs API routes — create, list, poll, and cancel perturbation jobs."""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from src.api.models import (
    AttackMethod, AttackParams, CreateJobRequest, JobListResponse,
    JobResponse, JobStatus, TargetTool, TrafficSource,
)
from src.tasks.celery_app import celery_app
from src.tasks.worker import run_perturbation_job

router = APIRouter()

# In-memory job store (replace with PostgreSQL in production)
_jobs: dict[str, dict] = {}

UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PCAP = Path("data/sample_pcaps/sample_traffic.pcap")


def _make_job(job_id: str, params: AttackParams, pcap_path: str, filename: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": now,
        "updated_at": now,
        "pcap_filename": filename,
        "pcap_path": pcap_path,
        "attack_params": params.dict(),
        "progress": 0,
        "progress_message": "Queued",
        "summary": None,
        "tool_results": None,
        "perturbed_pcap_url": None,
        "report_url": None,
        "error": None,
    }


def _to_response(job: dict) -> JobResponse:
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        pcap_filename=job.get("pcap_filename"),
        attack_params=AttackParams(**job["attack_params"]),
        progress=job.get("progress", 0),
        progress_message=job.get("progress_message"),
        summary=job.get("summary"),
        tool_results=job.get("tool_results"),
        perturbed_pcap_url=job.get("perturbed_pcap_url"),
        report_url=job.get("report_url"),
        error=job.get("error"),
    )


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    background_tasks: BackgroundTasks,
    traffic_source: TrafficSource = Form(TrafficSource.UPLOAD),
    method: AttackMethod = Form(AttackMethod.PGD),
    epsilon: float = Form(0.03),
    alpha: float = Form(0.005),
    iterations: int = Form(10),
    target_tools: str = Form("all"),          # comma-separated or "all"
    live_interface: Optional[str] = Form(None),
    capture_duration_s: int = Form(30),
    description: Optional[str] = Form(None),
    pcap_file: Optional[UploadFile] = File(None),
):
    """
    Create a new perturbation job.

    - Upload a PCAP or CSV file, choose a live interface, or use the built-in sample.
    - Tune epsilon, alpha, iterations.
    - Select target tools (comma-separated IDs or "all").
    """
    job_id = str(uuid.uuid4())

    # Parse target tools
    if target_tools.strip().lower() == "all":
        selected_tools = list(TargetTool)
    else:
        selected_tools = [TargetTool(t.strip()) for t in target_tools.split(",")]

    params = AttackParams(
        method=method,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        target_tools=selected_tools,
    )

    # Resolve input PCAP
    if traffic_source == TrafficSource.UPLOAD:
        if not pcap_file:
            raise HTTPException(status_code=400, detail="pcap_file required for upload source")
        dest = UPLOAD_DIR / f"{job_id}_{pcap_file.filename}"
        with open(dest, "wb") as f:
            content = await pcap_file.read()
            f.write(content)
        pcap_path = str(dest)
        filename = pcap_file.filename

    elif traffic_source == TrafficSource.SAMPLE:
        if not SAMPLE_PCAP.exists():
            raise HTTPException(status_code=404, detail="Sample PCAP not found. Run: python data/sample_pcaps/generate_sample_pcap.py")
        pcap_path = str(SAMPLE_PCAP)
        filename = SAMPLE_PCAP.name

    elif traffic_source == TrafficSource.LIVE:
        if not live_interface:
            raise HTTPException(status_code=400, detail="live_interface required for live capture")
        pcap_path = f"live://{live_interface}:{capture_duration_s}"
        filename = f"live_{live_interface}.pcap"

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported traffic_source: {traffic_source}")

    job = _make_job(job_id, params, pcap_path, filename)
    _jobs[job_id] = job

    # Dispatch to Celery worker
    background_tasks.add_task(
        _dispatch_celery,
        job_id=job_id,
        pcap_path=pcap_path,
        params_dict=params.dict(),
    )

    return _to_response(job)


async def _dispatch_celery(job_id: str, pcap_path: str, params_dict: dict):
    """Dispatch job to Celery and update in-memory store as result arrives."""
    _jobs[job_id]["status"] = JobStatus.RUNNING
    _jobs[job_id]["progress"] = 5
    _jobs[job_id]["progress_message"] = "Worker picked up job"
    _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        result = run_perturbation_job.delay(job_id, pcap_path, params_dict)
        task_result = result.get(timeout=300)          # wait up to 5 min
        _jobs[job_id].update(task_result)
        _jobs[job_id]["status"] = JobStatus.SUCCESS
        _jobs[job_id]["progress"] = 100
    except Exception as exc:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error"] = str(exc)
    finally:
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


@router.get("/", response_model=JobListResponse)
async def list_jobs(page: int = 1, page_size: int = 20):
    """List all perturbation jobs (paginated)."""
    all_jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    total = len(all_jobs)
    start = (page - 1) * page_size
    page_jobs = all_jobs[start : start + page_size]
    return JobListResponse(
        jobs=[_to_response(j) for j in page_jobs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get status and results for a specific job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _to_response(_jobs[job_id])


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _jobs[job_id]
    if job["status"] in (JobStatus.SUCCESS, JobStatus.FAILED):
        raise HTTPException(status_code=409, detail="Cannot cancel a completed job")
    _jobs[job_id]["status"] = JobStatus.FAILED
    _jobs[job_id]["error"] = "Cancelled by user"
    _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


@router.get("/{job_id}/pcap")
async def download_perturbed_pcap(job_id: str):
    """Download the perturbed PCAP file for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _jobs[job_id]
    if job["status"] != JobStatus.SUCCESS:
        raise HTTPException(status_code=409, detail="Job not yet complete")
    pcap_path = RESULTS_DIR / f"{job_id}_perturbed.pcap"
    if not pcap_path.exists():
        raise HTTPException(status_code=404, detail="Perturbed PCAP file not found")
    return FileResponse(str(pcap_path), filename=f"ape-bait-{job_id[:8]}.pcap", media_type="application/octet-stream")


@router.get("/{job_id}/report")
async def download_report(job_id: str):
    """Download the JSON validation report for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _jobs[job_id]
    if job["status"] != JobStatus.SUCCESS:
        raise HTTPException(status_code=409, detail="Job not yet complete")
    report_path = RESULTS_DIR / f"{job_id}_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(report_path), filename=f"ape-bait-report-{job_id[:8]}.json", media_type="application/json")
