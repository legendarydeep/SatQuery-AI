"""
API v1 — Execution Trace & SSE Streaming Endpoint (Critique 11)
Provides static trace logs and resilient live SSE streaming with Last-Event-ID replay.
"""

import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from satquery.agent.trace import trace_store
from satquery.api.v1.endpoints.query import _JOBS_DB

router = APIRouter(prefix="/jobs", tags=["Execution Trace"])


@router.get("/{job_id}/trace")
def get_job_trace(job_id: str):
    """Retrieves the complete static list of execution trace steps."""
    job = _JOBS_DB.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    run_id = job.get("run_id")
    if not run_id:
        return {"job_id": job_id, "status": job["status"], "events": []}

    events = trace_store.get_events(run_id)
    return {
        "job_id": job_id,
        "run_id": run_id,
        "total_events": len(events),
        "events": [ev.model_dump() for ev in events]
    }


@router.get("/{job_id}/trace/stream")
async def stream_job_trace(
    job_id: str,
    request: Request,
    last_event_id: Optional[int] = Header(None, alias="Last-Event-ID")
):
    """
    Live Server-Sent Events (SSE) stream for Deep's investigation-mode UI.
    Supports standard Last-Event-ID replay so no events are lost if the client connects late.
    """
    job = _JOBS_DB.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        current_id = last_event_id or 0
        while True:
            # Check client disconnect
            if await request.is_disconnected():
                break

            run_id = job.get("run_id")
            if run_id:
                new_events = trace_store.get_events_since(run_id, current_id)
                for ev in new_events:
                    current_id = ev.event_id
                    data = json.dumps(ev.model_dump())
                    yield f"id: {ev.event_id}\nevent: trace_step\ndata: {data}\n\n"

            # Check if job is completed or failed
            if job.get("status") in ["completed", "failed"]:
                yield f"event: job_done\ndata: {json.dumps({'status': job.get('status')})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
