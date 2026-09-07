"""
API v1 — Query & Jobs Endpoint (Critiques 10, 23, 32)
Handles natural language query dispatch, idempotency caching, offline demo mode, and asynchronous job execution.
"""

import os
import uuid
import hashlib
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from satquery.agent.orchestrator import orchestrator
from satquery.api.v1.endpoints.missions import _MISSIONS_DB
from satquery.api.v1.endpoints.assets import _ASSETS_METADATA

router = APIRouter(tags=["Query & Jobs"])

# Persistent job store
_JOBS_DB: Dict[str, Dict[str, Any]] = {}
# Idempotency cache: fingerprint -> job_id
_IDEMPOTENCY_CACHE: Dict[str, str] = {}


class QuerySubmissionRequest(BaseModel):
    query: str = Field(..., description="Natural language prompt")
    asset_ids: Optional[List[str]] = Field(None, description="Explicit list of asset IDs to analyze")
    aoi_geojson: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    job_id: str
    mission_id: Optional[str] = None
    status: str  # "queued", "running", "completed", "failed"
    run_id: Optional[str] = None
    is_demo_mode: bool = False


async def _run_job_worker(job_id: str, query: str, asset_paths: Dict[str, str], mission_id: Optional[str]):
    _JOBS_DB[job_id]["status"] = "running"
    try:
        # Check offline demo mode flag (Critique 32)
        if os.getenv("SATQUERY_DEMO_MODE", "false").lower() == "true":
            # Load offline fixture
            fixture_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "demo", "fallback_cache", "act2_bitemporal_change.json"
            )
            fixture_path = os.path.abspath(fixture_path)
            if os.path.exists(fixture_path):
                with open(fixture_path, "r") as f:
                    cached_data = json.load(f)
                _JOBS_DB[job_id]["status"] = "completed"
                _JOBS_DB[job_id]["result"] = cached_data
                _JOBS_DB[job_id]["is_demo_mode"] = True
                return

        # Normal adaptive agentic execution
        result = await orchestrator.run_query(
            query=query,
            asset_paths=asset_paths,
            mission_id=mission_id
        )
        _JOBS_DB[job_id]["status"] = "completed"
        _JOBS_DB[job_id]["result"] = result
        _JOBS_DB[job_id]["run_id"] = result.get("run_id")

        # Update mission context if mission_id provided
        if mission_id and mission_id in _MISSIONS_DB:
            mission = _MISSIONS_DB[mission_id]
            mission.last_query_id = job_id
            mission.query_history.append({
                "query": query,
                "job_id": job_id,
                "answer": result.get("answer")
            })
            if result.get("geojson_geometry"):
                mission.last_aoi_geojson = result["geojson_geometry"]

    except Exception as e:
        _JOBS_DB[job_id]["status"] = "failed"
        _JOBS_DB[job_id]["error"] = str(e)


@router.post("/missions/{mission_id}/query", response_model=JobStatusResponse)
async def submit_query(
    mission_id: str,
    request: QuerySubmissionRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    """
    Submits a query to the agentic planner for the given mission.
    Idempotent: duplicate requests return the existing job.
    """
    # 1. Resolve active assets
    asset_ids = request.asset_ids or []
    if not asset_ids and mission_id in _MISSIONS_DB:
        asset_ids = _MISSIONS_DB[mission_id].active_asset_ids

    asset_paths: Dict[str, str] = {}
    for idx, aid in enumerate(asset_ids):
        meta = _ASSETS_METADATA.get(aid)
        if meta and os.path.exists(meta["file_path"]):
            key = f"t{idx+1}" if len(asset_ids) > 1 else "primary"
            asset_paths[key] = meta["file_path"]

    # 2. Idempotency Check (Critique 23)
    fingerprint_material = f"{mission_id}:{request.query}:{sorted(asset_ids)}"
    fingerprint = idempotency_key or hashlib.sha256(fingerprint_material.encode("utf-8")).hexdigest()

    if fingerprint in _IDEMPOTENCY_CACHE:
        cached_job_id = _IDEMPOTENCY_CACHE[fingerprint]
        if cached_job_id in _JOBS_DB:
            job_info = _JOBS_DB[cached_job_id]
            return JobStatusResponse(
                job_id=cached_job_id,
                mission_id=mission_id,
                status=job_info["status"],
                run_id=job_info.get("run_id")
            )

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    _IDEMPOTENCY_CACHE[fingerprint] = job_id
    _JOBS_DB[job_id] = {
        "job_id": job_id,
        "mission_id": mission_id,
        "query": request.query,
        "status": "queued",
        "result": None,
        "error": None,
        "run_id": None
    }

    # Dispatch to background task worker
    background_tasks.add_task(
        _run_job_worker,
        job_id=job_id,
        query=request.query,
        asset_paths=asset_paths,
        mission_id=mission_id
    )

    return JobStatusResponse(
        job_id=job_id,
        mission_id=mission_id,
        status="queued"
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Polls async job status."""
    job = _JOBS_DB.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job_id,
        mission_id=job.get("mission_id"),
        status=job["status"],
        run_id=job.get("run_id"),
        is_demo_mode=job.get("is_demo_mode", False)
    )


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Retrieves final result, deterministic measurements, confidence, and GeoJSON."""
    job = _JOBS_DB.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error')}")
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Job still {job['status']}.")
    return job["result"]
