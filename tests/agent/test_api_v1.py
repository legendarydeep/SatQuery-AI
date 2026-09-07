"""
Integration tests for SatQuery API v1 Suite (Workstream E & Platform)
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import rasterio
from rasterio.transform import from_origin

from satquery.api.app import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_geotiff_bytes():
    """Generates an in-memory valid GeoTIFF for upload tests."""
    data = np.ones((1, 64, 64), dtype=np.uint8) * 100
    transform = from_origin(76.0, 11.0, 10.0, 10.0)
    crs = "EPSG:4326"

    bio = io.BytesIO()
    with rasterio.open(
        bio,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)
    bio.seek(0)
    return bio.getvalue()


def test_asset_upload_and_inspection(sample_geotiff_bytes):
    response = client.post(
        "/api/v1/assets/upload",
        files={"file": ("test_optical.tif", sample_geotiff_bytes, "image/tiff")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "asset_id" in data
    assert data["width"] == 64
    assert data["height"] == 64
    assert "EPSG:4326" in data["crs"]


def test_asset_upload_invalid_magic_bytes():
    fake_content = b"NOT_A_TIFF_HEADER"
    response = client.post(
        "/api/v1/assets/upload",
        files={"file": ("fake.tif", fake_content, "image/tiff")}
    )
    assert response.status_code == 400
    assert "Magic byte" in response.json()["detail"]


def test_mission_lifecycle_and_query_dispatch(sample_geotiff_bytes):
    # 1. Upload asset
    up_resp = client.post(
        "/api/v1/assets/upload",
        files={"file": ("mission_tile.tif", sample_geotiff_bytes, "image/tiff")}
    )
    asset_id = up_resp.json()["asset_id"]

    # 2. Create mission
    m_resp = client.post(
        "/api/v1/missions",
        json={"name": "Kerala Flood Mission", "description": "Monitoring water expansion"}
    )
    assert m_resp.status_code == 200
    mission_id = m_resp.json()["mission_id"]

    # 3. Attach asset
    att_resp = client.post(f"/api/v1/missions/{mission_id}/assets/{asset_id}")
    assert att_resp.status_code == 200

    # 4. Dispatch query with Idempotency-Key
    q_resp = client.post(
        f"/api/v1/missions/{mission_id}/query",
        json={"query": "Detect water spread across the tile", "asset_ids": [asset_id]},
        headers={"Idempotency-Key": "test_key_12345"}
    )
    assert q_resp.status_code == 200
    job_id = q_resp.json()["job_id"]

    # 5. Verify Idempotency: duplicate request returns identical job_id
    dup_resp = client.post(
        f"/api/v1/missions/{mission_id}/query",
        json={"query": "Detect water spread across the tile", "asset_ids": [asset_id]},
        headers={"Idempotency-Key": "test_key_12345"}
    )
    assert dup_resp.json()["job_id"] == job_id

    # 6. Check job status endpoint
    poll_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["job_id"] == job_id


def test_trace_endpoint():
    # Verify trace endpoint for a dummy job
    resp = client.get("/api/v1/jobs/dummy_job_999/trace")
    assert resp.status_code == 404
