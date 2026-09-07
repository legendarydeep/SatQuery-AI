"""
SatQuery API v1 Router Hub (Workstream E & Platform)
Mounts /assets, /missions, /query, /jobs, /trace, and /export under /api/v1 prefix.
"""

from fastapi import APIRouter
from satquery.api.v1.endpoints.assets import router as assets_router
from satquery.api.v1.endpoints.missions import router as missions_router
from satquery.api.v1.endpoints.query import router as query_router
from satquery.api.v1.endpoints.trace import router as trace_router
from satquery.api.v1.endpoints.export import router as export_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(assets_router)
api_v1_router.include_router(missions_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(trace_router)
api_v1_router.include_router(export_router)

__all__ = ["api_v1_router"]
