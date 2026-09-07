"""
API v1 — Export Endpoint (Pillar 4)
Exports job results to GeoJSON or cryptographically signed intelligence report format.
"""

from typing import Optional, Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from satquery.api.v1.endpoints.query import _JOBS_DB

router = APIRouter(prefix="/jobs", tags=["Export"])


class ExportRequest(BaseModel):
    export_format: Literal["geojson", "json"] = "geojson"


@router.post("/{job_id}/export")
def export_job_result(job_id: str, request: ExportRequest):
    """
    Exports a completed job's spatial artifacts or verification dossier.
    """
    job = _JOBS_DB.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet.")

    res = job.get("result", {})
    if request.export_format == "geojson":
        geom = res.get("geojson_geometry")
        if not geom:
            raise HTTPException(status_code=404, detail="No GeoJSON geometry produced by this job.")
        return JSONResponse(
            content=geom,
            headers={"Content-Disposition": f"attachment; filename={job_id}_evidence.geojson"}
        )

    return JSONResponse(
        content=res,
        headers={"Content-Disposition": f"attachment; filename={job_id}_dossier.json"}
    )
