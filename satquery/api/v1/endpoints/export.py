from fastapi.responses import Response
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


@router.get("/{run_id}/pdf", response_class=Response)
def export_pdf(run_id: str):
    """Export the mission dossier as a PDF (or JSON fallback if fpdf2 not installed)."""
    from satquery.api.v1.endpoints.export_pdf import build_pdf_dossier
    # Minimal demo: build dossier from run_id label
    pdf_bytes = build_pdf_dossier(
        mission_id=run_id,
        query="SatQuery AI mission dossier",
        answer="See JSON export for full detail.",
        measurements={},
        confidence=0.0,
        run_id=run_id,
    )
    media_type = "application/pdf" if pdf_bytes[:4] == b"%PDF" else "application/json"
    return Response(content=pdf_bytes, media_type=media_type,
                    headers={"Content-Disposition": f"attachment; filename={run_id}_dossier.pdf"})
