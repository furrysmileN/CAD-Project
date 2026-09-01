"""Standalone FastAPI application for HarnessCAD v1 and episode v2."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .compare_api import router as compare_router
from .harness_api import HARNESS_RUNS_DIR, router as harness_router
from .harness_api_v2 import HARNESS_RUNS_V2_DIR, TRACE_VERSION, router as harness_v2_router


app = FastAPI(
    title="HarnessCAD",
    version="2.0.0",
    description="Constrained CAD plan validation, execution, tracing, and artifact recording.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/harness-runs", StaticFiles(directory=HARNESS_RUNS_DIR), name="harness-runs")
app.mount("/harness-runs-v2", StaticFiles(directory=HARNESS_RUNS_V2_DIR), name="harness-runs-v2")
app.include_router(harness_router)
app.include_router(harness_v2_router)
app.include_router(compare_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "HarnessCAD", "traceVersion": TRACE_VERSION}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "HarnessCAD",
        "health": "/api/health",
        "docs": "/docs",
        "recommendedApi": "/api/harness-v2",
    }
