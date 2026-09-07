"""
satquery.api.app
================
FastAPI application exposing the SatQuery AI Change Detection pipeline
over HTTP.

Endpoints
---------
GET  /health
    Liveness probe — returns 200 OK with version info.

POST /api/validate-inputs
    Upload two GeoTIFF files and run GeoValidator checks only (fast).
    Returns validation summary without running the full pipeline.

POST /api/change-detection
    Upload two GeoTIFF files, optional metadata, and run the full
    bi-temporal change detection pipeline.
    Returns the structured JSON result including GeoJSON change mask,
    confidence score, execution trace, and summary text.

All endpoints are CORS-enabled for local frontend development.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from satquery import __version__
from satquery.change_detection.pipeline import ChangeDetector
from satquery.core.validator import validate_input_pair, ValidationError
from satquery.api.v1 import api_v1_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SatQuery AI — Change Detection API",
    description=(
        "Bi-temporal satellite change detection for ISRO Cartosat-2S/RISAT imagery. "
        "Supports optical and SAR inputs with pseudo-change suppression, "
        "confidence scoring, and GeoJSON output."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Suite (Workstream E & Platform)
app.include_router(api_v1_router)

# Temp directory for uploaded files (cleaned up after each request)
_TMP_ROOT = os.path.join(tempfile.gettempdir(), "satquery_uploads")
os.makedirs(_TMP_ROOT, exist_ok=True)


# ---------------------------------------------------------------------------
# Response schemas (Pydantic)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: float


class ValidationResponse(BaseModel):
    status: str
    t1_info: dict
    t2_info: dict
    warnings: list[str]


class ChangeDetectionResponse(BaseModel):
    status: str
    primary_index: str
    change_direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: str
    area_metrics: dict
    n_changed_pixels: int
    n_regions: int
    otsu_threshold: float
    n_pseudo_removed: int
    summary: str
    geojson: dict
    execution_trace: list[dict]
    warnings: list[str]
    sensor_calibration_note: str
    total_processing_ms: float


# ---------------------------------------------------------------------------
# Utility: save uploaded file to a temp path
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile, suffix: str = ".tif") -> str:
    """Save an uploaded file to a temp path and return the path."""
    tmp_path = os.path.join(_TMP_ROOT, f"{int(time.time() * 1000)}_{upload.filename or 'upload'}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(version=__version__, timestamp=time.time())


@app.post(
    "/api/validate-inputs",
    response_model=ValidationResponse,
    tags=["Validation"],
    summary="Validate a T1/T2 GeoTIFF pair (fast, no change detection).",
)
async def validate_inputs(
    file_t1: Annotated[UploadFile, File(description="T1 (earlier) GeoTIFF")],
    file_t2: Annotated[UploadFile, File(description="T2 (later)   GeoTIFF")],
    sensor_t1: Annotated[str, Form()] = "optical",
    sensor_t2: Annotated[str, Form()] = "optical",
    timestamp_t1: Annotated[str | None, Form()] = None,
    timestamp_t2: Annotated[str | None, Form()] = None,
) -> ValidationResponse:
    """
    Upload two GeoTIFFs and run all GeoValidator checks.
    Returns a summary without running the full change detection pipeline.
    """
    path_t1 = path_t2 = ""
    try:
        path_t1 = _save_upload(file_t1)
        path_t2 = _save_upload(file_t2)

        warnings: list[str] = []
        try:
            info_t1, info_t2 = validate_input_pair(
                path_t1, path_t2,
                sensor_t1=sensor_t1,
                sensor_t2=sensor_t2,
                timestamp_t1=timestamp_t1,
                timestamp_t2=timestamp_t2,
            )
            status = "ok"
        except ValidationError as exc:
            info_t1 = info_t2 = {}
            warnings.append(str(exc))
            status = "error"

        # Convert non-serialisable rasterio objects to strings
        def _serialise(info: dict) -> dict:
            return {
                k: str(v) if not isinstance(v, (str, int, float, type(None))) else v
                for k, v in info.items()
            }

        return ValidationResponse(
            status=status,
            t1_info=_serialise(info_t1),
            t2_info=_serialise(info_t2),
            warnings=warnings,
        )
    finally:
        _cleanup(path_t1, path_t2)


@app.post(
    "/api/change-detection",
    tags=["Change Detection"],
    summary="Run full bi-temporal change detection pipeline.",
)
async def change_detection(
    file_t1: Annotated[UploadFile, File(description="T1 (earlier) GeoTIFF")],
    file_t2: Annotated[UploadFile, File(description="T2 (later)   GeoTIFF")],
    sensor_t1:         Annotated[str,        Form()] = "optical",
    sensor_t2:         Annotated[str,        Form()] = "optical",
    timestamp_t1:      Annotated[str | None, Form()] = None,
    timestamp_t2:      Annotated[str | None, Form()] = None,
    query_hint:        Annotated[str,        Form()] = "",
    suppress_pseudo:   Annotated[bool,       Form()] = True,
    smooth_sigma:      Annotated[float,      Form()] = 1.5,
    open_radius:       Annotated[int,        Form()] = 2,
    close_radius:      Annotated[int,        Form()] = 3,
    min_area_px:       Annotated[int,        Form()] = 25,
) -> JSONResponse:
    """
    Upload two GeoTIFFs and run the full SatQuery AI change detection pipeline.

    Parameters (form fields)
    ------------------------
    sensor_t1 / sensor_t2 : "optical" | "sar" | "cartosat" | "risat" | "sentinel"
    timestamp_t1 / t2     : ISO-8601 dates (optional, e.g. "2023-01-15")
    query_hint             : Free text from user query ("flood", "deforestation", …)
    suppress_pseudo        : Enable STSF-Net pseudo-change suppression (default True)
    smooth_sigma           : Gaussian blur sigma in pixels (default 1.5)
    open_radius            : Morphological opening radius (default 2)
    close_radius           : Morphological closing radius (default 3)
    min_area_px            : Minimum connected-component area in pixels (default 25)
    """
    path_t1 = path_t2 = ""
    try:
        path_t1 = _save_upload(file_t1)
        path_t2 = _save_upload(file_t2)

        detector = ChangeDetector(
            sensor_t1=sensor_t1,
            sensor_t2=sensor_t2,
            query_hint=query_hint,
            suppress_pseudo=suppress_pseudo,
            smooth_sigma=smooth_sigma,
            open_radius=open_radius,
            close_radius=close_radius,
            min_area_px=min_area_px,
        )

        result = detector.run_from_files(
            path_t1, path_t2,
            timestamp_t1=timestamp_t1,
            timestamp_t2=timestamp_t2,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result)

        # Convert numpy types for JSON serialisation
        return JSONResponse(content=_jsonify(result))

    finally:
        _cleanup(path_t1, path_t2)


# ---------------------------------------------------------------------------
# JSON serialisation helper (handle numpy scalars)
# ---------------------------------------------------------------------------

def _jsonify(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
