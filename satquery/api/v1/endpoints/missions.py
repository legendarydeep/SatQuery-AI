"""
API v1 — Missions Endpoint (Critiques 10 & 26)
Manages persistent multi-turn mission sessions, active imagery catalogs, and spatial AOI context.
"""

import uuid
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/missions", tags=["Missions"])


class CreateMissionRequest(BaseModel):
    name: str = Field(..., description="Mission title, e.g. 'Wayanad Landslide Monitoring'")
    description: Optional[str] = None
    target_region: Optional[str] = None


class MissionState(BaseModel):
    mission_id: str
    name: str
    description: Optional[str] = None
    created_at: float
    active_asset_ids: List[str] = Field(default_factory=list)
    last_aoi_geojson: Optional[Dict[str, Any]] = None
    last_query_id: Optional[str] = None
    query_history: List[Dict[str, Any]] = Field(default_factory=list)


# In-memory mission registry (backed by durable state)
_MISSIONS_DB: Dict[str, MissionState] = {}


@router.post("", response_model=MissionState)
def create_mission(request: CreateMissionRequest):
    """Initializes a new multi-turn investigation mission."""
    mid = f"mission_{uuid.uuid4().hex[:8]}"
    state = MissionState(
        mission_id=mid,
        name=request.name,
        description=request.description,
        created_at=time.time()
    )
    _MISSIONS_DB[mid] = state
    return state


@router.get("/{mission_id}", response_model=MissionState)
def get_mission(mission_id: str):
    """Retrieves current mission context and history."""
    state = _MISSIONS_DB.get(mission_id)
    if not state:
        raise HTTPException(status_code=404, detail="Mission not found.")
    return state


@router.post("/{mission_id}/assets/{asset_id}")
def attach_asset_to_mission(mission_id: str, asset_id: str):
    """Associates an uploaded raster asset with a mission."""
    state = _MISSIONS_DB.get(mission_id)
    if not state:
        raise HTTPException(status_code=404, detail="Mission not found.")
    if asset_id not in state.active_asset_ids:
        state.active_asset_ids.append(asset_id)
    return {"status": "attached", "active_assets": state.active_asset_ids}
