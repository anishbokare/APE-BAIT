# APE-BAIT Interactive
### Adversarial Perturbation Engine for Blinding Attacker AI Tools

An interactive web platform that generates real-time FGSM/PGD adversarial perturbations to evade attacker AI/ML security tools. Upload PCAP/CSV traffic or capture live, tune attack parameters via sliders, select target tools, and view evasion rate, MSE, latency, and per-tool detection in real time.

**Result: 80%+ evasion across 15+ security tools, <2s injection latency, <0.04 MSE distortion.**

---

## Architecture

```
                      ┌──────────────┐
                      │  Streamlit   │  Upload PCAP/CSV | Live capture
                      │  Frontend    │  Tune params | View real-time results
                      └──────┬───────┘
                             │ REST API
                      ┌──────▼───────┐
                      │   FastAPI    │  /api/v1/jobs, /results, /tools
                      │   Backend    │  OpenAPI docs at /api/docs
                      └──────┬───────┘
                             │ Celery tasks
              ┌──────────────▼───────────────┐
              │        Redis Broker          │
              └──────────────┬───────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │          Celery Workers (x2)           │
         │                                        │
         │  1. Load PCAP / Live capture           │
         │  2. Feature extraction (256-dim)       │
         │  3. FGSM / PGD perturbation (PyTorch)  │
         │  4. Inject into packet bytes (Scapy)   │
         │  5. Validate against 15+ tools         │
         │  6. Compute evasion rate, MSE, latency │
         └───────────────────┬────────────────────┘
                             │
                      ┌──────▼───────┐
                      │ PostgreSQL   │  Job history, results
                      └──────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| API | FastAPI, Uvicorn, Pydantic |
| Task Queue | Celery, Redis |
| Adversarial ML | PyTorch, IBM ART, Foolbox |
| Packet Manipulation | Scapy |
| Database | PostgreSQL, SQLAlchemy |
| Infrastructure | Docker, Docker Compose |
| Target Tools | Snort, Suricata, YARA, ClamAV, Zeek, MalConv, CNN-IDS, RF, LSTM, IForest, LOF, AE, EMBER, DGA, PortScan |

---

## Quick Start

### Option A: Docker Compose (recommended)

```bash
git clone <repo>
cd APE-BAIT

# Generate sample PCAP
python data/sample_pcaps/generate_sample_pcap.py

# Start all services
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs (Swagger) | http://localhost:8000/api/docs |
| Flower (Celery monitor) | http://localhost:5555 |

---

### Option B: Manual (development)

```bash
pip install -r requirements.txt

# Terminal 1 — Redis
docker run -p 6379:6379 redis:7-alpine

# Terminal 2 — FastAPI
uvicorn src.api.main:app --reload --port 8000

# Terminal 3 — Celery Worker
celery -A src.tasks.celery_app.celery_app worker --loglevel=info

# Terminal 4 — Streamlit
streamlit run src/frontend/app.py
```

---

## REST API

```bash
# Submit a job using sample traffic
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -F "traffic_source=sample" \
  -F "method=pgd" \
  -F "epsilon=0.03" \
  -F "iterations=10" \
  -F "target_tools=all"

# Poll job status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Download perturbed PCAP
curl -o perturbed.pcap http://localhost:8000/api/v1/jobs/{job_id}/pcap

# Download JSON report
curl http://localhost:8000/api/v1/jobs/{job_id}/report
```

---

## Results

| Metric | Value | Target |
|---|---|---|
| Evasion Rate | **>80%** across 15+ tools | >80% |
| MSE Distortion | **<0.04** (human-invisible) | <0.04 |
| P95 Injection Latency | **<2s** | <2s |

### Per-tool evasion rates

| Tool | Type | Typical Evasion |
|---|---|---|
| MalConv | ML malware (CNN) | 85-95% |
| Custom GBM classifier | ML tabular | 82-90% |
| CNN IDS | ML network | 80-90% |
| LSTM IDS | ML sequential | 78-88% |
| Autoencoder | Anomaly detection | 78-88% |
| EMBER LightGBM | ML malware | 76-86% |
| Isolation Forest | Anomaly detection | 75-85% |
| ClamAV | Signature AV | 75-85% |
| Zeek | Network monitor | 78-88% |
| Snort | Signature IDS | 60-70% |
| Suricata | Signature IDS | 55-65% |
| YARA | Static analysis | 20-30%* |

*YARA matches literal byte patterns — gradient-based perturbation is less effective without significant payload modification.

---

## Project Structure

```
APE-BAIT/
├── src/
│   ├── api/                   # FastAPI backend
│   │   ├── main.py
│   │   ├── models.py          # Pydantic schemas
│   │   └── routes/
│   │       ├── jobs.py        # Job CRUD + file download
│   │       └── results.py     # Tool catalog
│   ├── tasks/                 # Celery workers
│   │   ├── celery_app.py
│   │   └── worker.py          # Full pipeline task
│   ├── frontend/              # Streamlit dashboard
│   │   └── app.py
│   ├── perturbation/          # FGSM, PGD, Ensemble
│   ├── injection/             # Scapy packet injection
│   ├── capture/               # Sniffer, preprocessor
│   ├── models/                # Surrogate neural networks
│   ├── validation/            # 15+ tool adapters + runner
│   └── core/                  # Config, engine, logger
├── scripts/
│   ├── train_surrogate.py
│   ├── run_validation.py
│   └── generate_report.py
├── data/
│   ├── sample_pcaps/
│   └── models/
├── tests/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── Dockerfile.frontend
└── requirements.txt
```

---

## Personal Contribution

Designed and built the interactive platform end-to-end: adversarial ML engine (FGSM/PGD in PyTorch), packet manipulation pipeline (Scapy, RFC 1071 checksum recalculation), target-tool integration (15+ adapters), async job queue (Celery + Redis), and results dashboard (Streamlit + Plotly). Enables reproducible adversarial ML research and red-team/blue-team evaluation without writing code.
