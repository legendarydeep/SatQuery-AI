"""
satquery.core.band_mapper
=========================
Sensor-specific band mapping and canonical spectral channel alignment.
Decouples raw sensor band order from model tensor and index expectations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Canonical spectral band index layouts per sensor
SENSOR_BAND_PROFILES: Dict[str, Dict[str, int]] = {
    # Sentinel-2 MSI L2A canonical order: B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR)
    "sentinel2": {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4},
    "sentinel-2": {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4},
    
    # Landsat-8/9 OLI: B2(Blue), B3(Green), B4(Red), B5(NIR), B6(SWIR1)
    "landsat8": {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4},
    "landsat-8": {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4},
    
    # ISRO Cartosat-2S 4-band Multispectral: B1(Blue), B2(Green), B3(Red), B4(NIR)
    "cartosat": {"blue": 0, "green": 1, "red": 2, "nir": 3},
    "cartosat-2s": {"blue": 0, "green": 1, "red": 2, "nir": 3},
    
    # Dual-Pol SAR (RISAT-1C / Sentinel-1): VV=0, VH=1
    "sar": {"vv": 0, "vh": 1},
    "risat": {"vv": 0, "vh": 1},
    "risat-1c": {"vv": 0, "vh": 1},
}


class BandMapper:
    """
    Extracts and maps raw satellite bands to model-expected tensor layouts.
    """

    @staticmethod
    def get_profile(sensor_name: str) -> Dict[str, int]:
        sensor_key = sensor_name.lower().strip()
        for k, v in SENSOR_BAND_PROFILES.items():
            if k in sensor_key:
                return v
        # Generic optical default: Blue, Green, Red, NIR, SWIR
        return {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4}

    @staticmethod
    def extract_rgb(array: np.ndarray, sensor_name: str = "optical") -> np.ndarray:
        """
        Extract a 3-channel (Red, Green, Blue) normalized float32 array (3, H, W)
        tailored for deep learning backbones (e.g. ChangeFormer).
        """
        profile = BandMapper.get_profile(sensor_name)
        n_bands, h, w = array.shape

        r_idx = profile.get("red", 0 if n_bands == 1 else 2)
        g_idx = profile.get("green", 0 if n_bands == 1 else 1)
        b_idx = profile.get("blue", 0)

        # Bounds check
        r_idx = min(r_idx, n_bands - 1)
        g_idx = min(g_idx, n_bands - 1)
        b_idx = min(b_idx, n_bands - 1)

        rgb = np.stack([array[r_idx], array[g_idx], array[b_idx]], axis=0).astype(np.float32)

        # Min-max scaling to [0, 1] per channel
        for c in range(3):
            c_min, c_max = float(np.nanmin(rgb[c])), float(np.nanmax(rgb[c]))
            if c_max > c_min:
                rgb[c] = (rgb[c] - c_min) / (c_max - c_min)
            else:
                rgb[c] = 0.0

        return np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
