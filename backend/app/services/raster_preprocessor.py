"""
backend.app.services.raster_preprocessor
========================================
Converts satellite GeoTIFF and high-res imagery into standardized RGB PNGs
with percentile contrast stretching, tiling, and visual previews for VLM and detector ingestion.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from app.services.geo_compat import open_raster

logger = logging.getLogger(__name__)

TEMP_SATQUERY_DIR = Path(tempfile.gettempdir()) / "satquery"
TEMP_SATQUERY_DIR.mkdir(parents=True, exist_ok=True)


class RasterPreprocessor:
    """
    Standardized satellite raster preprocessor for RS VLMs, detectors, and segmenters.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or TEMP_SATQUERY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def percentile_stretch(
        self,
        array: np.ndarray,
        p_min: float = 2.0,
        p_max: float = 98.0,
    ) -> np.ndarray:
        """
        Clips extreme solar reflectance outliers (2nd to 98th percentile)
        and normalizes to [0, 255] uint8.
        """
        stretched = np.zeros_like(array, dtype=np.float32)
        if array.ndim == 2:
            channels = [array]
        else:
            channels = [array[c] for c in range(array.shape[0])]

        norm_channels = []
        for ch in channels:
            valid_pixels = ch[np.isfinite(ch)]
            if valid_pixels.size == 0:
                norm_channels.append(np.zeros_like(ch, dtype=np.uint8))
                continue
            vmin, vmax = np.percentile(valid_pixels, (p_min, p_max))
            if vmax > vmin:
                clipped = np.clip(ch, vmin, vmax)
                norm = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(ch, dtype=np.uint8)
            norm_channels.append(norm)

        if array.ndim == 2:
            return norm_channels[0]
        return np.stack(norm_channels, axis=0)

    def prepare_for_vlm(
        self,
        tif_path: str | Path,
        tile_size: int = 512,
        preview_size: Tuple[int, int] = (1024, 1024),
    ) -> Dict[str, Any]:
        """
        Ingests a GeoTIFF, extracts RGB channels, applies percentile normalization,
        generates an overview preview PNG, and creates non-overlapping square tiles.

        Returns:
            {
                "preview_path": Path to preview.png,
                "tile_paths": [Path to tile_001.png, ...],
                "width": original width,
                "height": original height,
                "bands": band count,
                "crs": crs string
            }
        """
        path = Path(tif_path)
        stem = path.stem

        with open_raster(path) as ds:
            width = ds.width
            height = ds.height
            count = ds.count
            crs = str(getattr(ds, "crs", "EPSG:4326") or "EPSG:4326")
            bounds = getattr(ds, "bounds", None)

            # Read top 3 bands (or duplicate single band for greyscale/SAR)
            if count >= 3:
                try:
                    data = ds.read([1, 2, 3])
                except Exception:
                    data = ds.read()[:3]
            else:
                band1 = ds.read(1)
                data = np.stack([band1, band1, band1], axis=0)

        # Apply 2-98% percentile stretch
        norm_chw = self.percentile_stretch(data, 2.0, 98.0)
        # Convert CHW -> HWC
        hwc = np.transpose(norm_chw, (1, 2, 0))
        img = Image.fromarray(hwc, mode="RGB")

        # 1. Save main preview.png
        preview_img = img.copy()
        preview_img.thumbnail(preview_size, Image.Resampling.LANCZOS)
        preview_path = self.output_dir / f"{stem}_preview.png"
        preview_img.save(preview_path, format="PNG", optimize=True)

        # 2. Tile high-res raster into standard chunks
        tile_paths: List[str] = []
        tile_idx = 1
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                bottom = min(top + tile_size, height)
                
                # Only keep tiles of sufficient area (> 25% full tile)
                if (right - left) >= 64 and (bottom - top) >= 64:
                    box = (left, top, right, bottom)
                    tile = img.crop(box)
                    t_path = self.output_dir / f"{stem}_tile_{tile_idx:03d}.png"
                    tile.save(t_path, format="PNG")
                    tile_paths.append(str(t_path))
                    tile_idx += 1

        return {
            "preview_path": str(preview_path),
            "tile_paths": tile_paths,
            "width": width,
            "height": height,
            "bands": count,
            "crs": crs,
            "tile_count": len(tile_paths),
        }


raster_preprocessor = RasterPreprocessor()
prepare_for_vlm = raster_preprocessor.prepare_for_vlm
