"""
satquery.change_detection.stsf_multimodal
=========================================
Spatio-Temporal Structural Feature (STSF) cross-modal pseudo-change suppression.
Dedicated to the Multimodal Optical-SAR change detection workstream.

Mitigates pseudo-changes caused by fundamentally different imaging physics
between optical surface reflectance and SAR microwave backscatter geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def stsf_multimodal_filter(
    diff_map: np.ndarray,
    optical_band: np.ndarray,
    sar_band: np.ndarray,
    window_size: int = 9,
    structural_threshold: float = 0.20,
) -> tuple[np.ndarray, int]:
    """
    Suppresses false alarms in Optical-SAR cross-modal difference maps.

    Parameters
    ----------
    diff_map : (H, W) float32
        Absolute difference between normalized optical and SAR indicators.
    optical_band : (H, W) float32
        Optical reflectance channel.
    sar_band : (H, W) float32
        SAR backscatter channel (calibrated amplitude or dB).
    window_size : int
        Local analysis window (pixels).
    structural_threshold : float
        Tolerance for common structural feature consistency.

    Returns
    -------
    (filtered_diff_map, pseudo_pixels_suppressed)
    """
    # 1. Local spatial mean
    mu_opt = uniform_filter(optical_band, size=window_size)
    mu_sar = uniform_filter(sar_band, size=window_size)

    # 2. Local variance
    var_opt = uniform_filter(optical_band**2, size=window_size) - mu_opt**2
    var_sar = uniform_filter(sar_band**2, size=window_size) - mu_sar**2
    std_opt = np.sqrt(np.maximum(var_opt, 0))
    std_sar = np.sqrt(np.maximum(var_sar, 0))

    # 3. Structural cross-correlation
    cov = uniform_filter(optical_band * sar_band, size=window_size) - mu_opt * mu_sar
    struct_sim = cov / (std_opt * std_sar + 1e-6)

    # Where structural similarity is high despite radiometric differences,
    # the apparent difference is an imaging-physics pseudo-change
    pseudo_mask = (struct_sim > (1.0 - structural_threshold)) & (diff_map > 0)

    filtered_diff = diff_map.copy()
    filtered_diff[pseudo_mask] = 0.0
    n_suppressed = int(pseudo_mask.sum())

    return filtered_diff, n_suppressed
