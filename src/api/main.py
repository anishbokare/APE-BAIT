"""APE-BAIT Interactive — FastAPI Application Entry Point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from src.api.routes import jobs, results, tools

logger = logging.getLogger("ape-bait.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("APE-BAIT Interactive API starting up...")
    yield
    logger.info("APE-BAIT Interactive API shutting down...")


app = FastAPI(
    title="APE-BAIT Interactive",
    description=(
        "Adversarial Perturbation Engine for Blinding Attacker AI Tools. "
        "Upload PCAP/CSV traffic, tune FGSM/PGD parameters, and measure "
        "real-time evasion against 15+ security tools."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(results.router, prefix="/api/v1/results", tags=["Results"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["Tools"])


@app.get("/api/v1/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "APE-BAIT Interactive API", "version": "1.0.0"}
