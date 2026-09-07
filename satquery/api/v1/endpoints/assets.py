"""
API v1 — Raster Assets Endpoint (Critique 24)
Handles GeoTIFF upload, MIME validation, decompression bomb defense, and metadata extraction.
"""

import os
import uuid
import shutil
import tempfile
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException
import rasterio

router = APIRouter(prefix="/assets", tags=["Assets"])

# 500MB maximum upload limit
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
# 50 Megapixel maximum uncompressed pixel limit
MAX_UNCOMPRESSED_PIXELS = 50_000_000

_ASSETS_DIR = os.path.join(tempfile.gettempdir(), "satquery_v1_assets")
os.makedirs(_ASSETS_DIR, exist_ok=True)

# In-memory registry of parsed assets
_ASSETS_METADATA: Dict[str, Dict[str, Any]] = {}


@router.post("/upload")
async def upload_asset(file: UploadFile = File(...)):
    """
    Uploads a satellite raster (GeoTIFF), validates magic bytes, prevents path traversal,
    and extracts geospatial metadata (CRS, bounds, resolution).
    """
    # 1. Path traversal defense: sanitize filename
    safe_filename = os.path.basename(file.filename or "uploaded.tif")
    if not safe_filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif, .tiff) files are supported.")

    asset_id = f"asset_{uuid.uuid4().hex[:12]}"
    dest_path = os.path.join(_ASSETS_DIR, f"{asset_id}_{safe_filename}")

    # 2. Stream to disk with size boundary
    total_bytes = 0
    with open(dest_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                os.remove(dest_path)
                raise HTTPException(status_code=413, detail=f"File exceeds maximum upload size of 500MB.")
            f.write(chunk)

    # 3. Magic bytes validation (TIFF magic: II\x2a\x00 or MM\x00\x2a)
    valid_magic = False
    with open(dest_path, "rb") as f:
        header = f.read(4)
        if header in (b"II*\x00", b"MM\x00*"):
            valid_magic = True
    if not valid_magic:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail="Invalid GeoTIFF format: Magic byte header mismatch.")

    # 4. Decompression bomb check & Metadata extraction via rasterio
    try:
        with rasterio.open(dest_path) as src:
            num_pixels = src.width * src.height * src.count
            bounds = src.bounds
            crs_str = src.crs.to_string() if src.crs else "UNKNOWN"
            res = src.res
            width, height, count = src.width, src.height, src.count

        if num_pixels > MAX_UNCOMPRESSED_PIXELS:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise HTTPException(
                status_code=400,
                detail=f"Decompression defense: Raster contains {num_pixels} pixels, exceeding 50M pixel safety limit."
            )

        meta = {
            "asset_id": asset_id,
            "filename": safe_filename,
            "file_path": dest_path,
            "file_size_bytes": total_bytes,
            "width": width,
            "height": height,
            "bands": count,
            "crs": crs_str,
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "resolution": [res[0], res[1]],
            "is_sar": "sar" in safe_filename.lower() or "s1" in safe_filename.lower()
        }
        _ASSETS_METADATA[asset_id] = meta
        return meta
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail=f"Failed to inspect GeoTIFF: {str(e)}")


@router.get("/{asset_id}")
def get_asset(asset_id: str):
    """Retrieves metadata of an uploaded raster asset."""
    meta = _ASSETS_METADATA.get(asset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Asset not found.")
    return meta
