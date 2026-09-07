"""
satquery.change_detection.metrics
===================================
Real-world area calculation, percentage change, and directional
change classification.

Change direction labels
------------------------
The pipeline maps spectral index changes to one of five semantic labels:

    vegetation_loss     — NDVI decreased significantly
    vegetation_gain     — NDVI increased significantly
    water_expansion     — NDWI increased significantly
    water_contraction   — NDWI decreased significantly
    urban_growth        — NDBI increased / NDVI decreased in non-water area
    sar_change          — Generic label for SAR-only change (no optical index)
    mixed_change        — Multiple indices contradict each other
    no_change           — Mask is essentially empty

Classification is heuristic / rule-based and explicitly designed to be
interpretable by non-expert users (problem statement requirement).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Semantic change labels
ChangeLabel = Literal[
    "vegetation_loss",
    "vegetation_gain",
    "water_expansion",
    "water_contraction",
    "urban_growth",
    "sar_change",
    "mixed_change",
    "no_change",
]

# Minimum fraction of changed pixels required to classify direction
_MIN_CHANGED_FRACTION = 0.001   # 0.1 %


class ChangeClassification(str):
    """
    Semantic change classification label with full provenance tracking.
    Inherits from str for seamless backward compatibility with all label equality checks,
    while carrying provenance metadata (classification_method, evidence, confidence).
    """
    change_type: str
    classification_method: str
    classification_evidence: list[str]
    classification_confidence: float

    def __new__(
        cls,
        label: str,
        classification_method: str = "spectral_rule",
        classification_evidence: list[str] | None = None,
        classification_confidence: float = 1.0,
    ):
        obj = super().__new__(cls, label)
        obj.change_type = label
        obj.classification_method = classification_method
        obj.classification_evidence = classification_evidence or []
        obj.classification_confidence = round(float(classification_confidence), 4)
        return obj

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "change_type":
                return self.change_type
            elif item == "classification_method":
                return self.classification_method
            elif item == "classification_evidence":
                return self.classification_evidence
            elif item == "classification_confidence":
                return self.classification_confidence
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "classification_method": self.classification_method,
            "classification_evidence": self.classification_evidence,
            "classification_confidence": self.classification_confidence,
        }


# ---------------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------------

def compute_area(
    mask: np.ndarray,
    transform,
) -> dict[str, float]:
    """
    Compute the real-world area of changed pixels using the raster's
    affine transform.

    Parameters
    ----------
    mask      : (H, W) bool array — True = changed pixel.
    transform : rasterio Affine transform (pixel size encoded in |a| and |e|).

    Returns
    -------
    dict with:
        "area_m2"          : float — total changed area in square metres.
        "area_ha"          : float — total changed area in hectares.
        "area_km2"         : float — total changed area in square kilometres.
        "n_changed_pixels" : int
        "total_pixels"     : int
        "pct_changed"      : float — percentage of image pixels that changed.
    """
    n_changed = int(mask.sum())
    total_px  = mask.size

    # Pixel ground sampling distance in map units (convert degrees to metres if in geographic CRS)
    gsd_x = abs(float(transform.a))
    gsd_y = abs(float(transform.e))
    if gsd_x < 0.05:  # Geographic degrees (e.g. EPSG:4326)
        _M_PER_DEG = 111_319.0
        pixel_area_m2 = (gsd_x * _M_PER_DEG) * (gsd_y * _M_PER_DEG)
    else:
        pixel_area_m2 = gsd_x * gsd_y

    area_m2  = n_changed * pixel_area_m2
    area_ha  = area_m2 / 10_000.0
    area_km2 = area_m2 / 1_000_000.0

    return {
        "area_m2":          round(area_m2, 2),
        "area_ha":          round(area_ha, 4),
        "area_km2":         round(area_km2, 6),
        "n_changed_pixels": n_changed,
        "total_pixels":     total_px,
        "pct_changed":      round(100.0 * n_changed / max(total_px, 1), 4),
    }


# ---------------------------------------------------------------------------
# Direction classification
# ---------------------------------------------------------------------------

def classify_change_direction(
    signed_diff_primary: np.ndarray,
    change_mask: np.ndarray,
    primary_index: str = "ndvi",
    signed_diff_secondary: np.ndarray | None = None,
    secondary_index: str | None = None,
) -> ChangeClassification:
    """
    Classify the dominant direction of change over the changed pixels.

    Parameters
    ----------
    signed_diff_primary   : (H, W) float32 signed difference for the primary index
                            (T2 - T1 > 0 means the index increased).
    change_mask           : (H, W) bool — which pixels changed.
    primary_index         : Index name ("ndvi", "ndwi", "ndbi", "rvi", …).
    signed_diff_secondary : Optional secondary index signed difference.
    secondary_index       : Name of the secondary index.

    Returns
    -------
    ChangeClassification (subclass of str) containing label, classification_method,
    classification_evidence, and classification_confidence.
    """
    pct_changed = change_mask.sum() / max(change_mask.size, 1)
    if pct_changed < _MIN_CHANGED_FRACTION:
        return ChangeClassification(
            "no_change",
            classification_method="spectral_rule",
            classification_evidence=["change_area_fraction_below_threshold"],
            classification_confidence=1.0,
        )

    # Mean signed change within the change mask
    changed_pixels = signed_diff_primary[change_mask]
    if changed_pixels.size == 0:
        return ChangeClassification(
            "no_change",
            classification_method="spectral_rule",
            classification_evidence=["zero_changed_pixels"],
            classification_confidence=1.0,
        )

    mean_delta = float(np.median(changed_pixels))  # median is more robust than mean
    label, evidence, confidence = _classify_single_index_with_evidence(primary_index, mean_delta)

    # Cross-check with secondary index if available
    if signed_diff_secondary is not None and secondary_index is not None:
        sec_pixels = signed_diff_secondary[change_mask]
        if sec_pixels.size > 0:
            sec_mean = float(np.median(sec_pixels))
            sec_label, sec_evidence, sec_conf = _classify_single_index_with_evidence(secondary_index, sec_mean)
            evidence.extend(sec_evidence)

            if sec_label != label and sec_label != "no_change":
                # Contradicting signals — flag as mixed
                logger.debug(
                    "Mixed change: primary=%s (%s) vs secondary=%s (%s)",
                    primary_index, label, secondary_index, sec_label,
                )
                label = "mixed_change"
                confidence = round((confidence + sec_conf) / 2.0 * 0.75, 4)

    return ChangeClassification(
        label,
        classification_method="spectral_rule",
        classification_evidence=evidence,
        classification_confidence=confidence,
    )


def classify_change_with_provenance(
    signed_diff_primary: np.ndarray,
    change_mask: np.ndarray,
    primary_index: str = "ndvi",
    signed_diff_secondary: np.ndarray | None = None,
    secondary_index: str | None = None,
) -> dict[str, Any]:
    """Convenience helper returning the classification provenance dictionary."""
    return classify_change_direction(
        signed_diff_primary=signed_diff_primary,
        change_mask=change_mask,
        primary_index=primary_index,
        signed_diff_secondary=signed_diff_secondary,
        secondary_index=secondary_index,
    ).to_dict()


def _classify_single_index_with_evidence(
    index_name: str, mean_delta: float
) -> tuple[ChangeLabel, list[str], float]:
    """Map (index, direction) to (ChangeLabel, evidence_list, confidence)."""
    idx = index_name.lower()
    delta_threshold = 0.02

    if abs(mean_delta) < delta_threshold:
        return "no_change", [f"{idx.upper()}_insignificant_delta ({mean_delta:+.4f})"], 0.5

    conf = min(1.0, round(float(abs(mean_delta) / 0.20), 4))
    conf = max(0.5, conf)

    if idx == "ndvi":
        if mean_delta > 0:
            return "vegetation_gain", [f"NDVI_increase (median_delta={mean_delta:+.4f})"], conf
        return "vegetation_loss", [f"NDVI_decrease (median_delta={mean_delta:+.4f})"], conf

    if idx == "ndwi":
        if mean_delta > 0:
            return "water_expansion", [f"NDWI_increase (median_delta={mean_delta:+.4f})"], conf
        return "water_contraction", [f"NDWI_decrease (median_delta={mean_delta:+.4f})"], conf

    if idx == "ndbi":
        if mean_delta > 0:
            return "urban_growth", [f"NDBI_increase (median_delta={mean_delta:+.4f})"], conf
        return "vegetation_gain", [f"NDBI_decrease (median_delta={mean_delta:+.4f})"], conf

    if idx in ("rvi", "db_vv", "db_vh"):
        return "sar_change", [f"SAR_{idx.upper()}_backscatter_shift (median_delta={mean_delta:+.4f})"], conf

    return "mixed_change", [f"generic_spectral_shift ({idx.upper()} delta={mean_delta:+.4f})"], 0.6


def _classify_single_index(index_name: str, mean_delta: float) -> ChangeLabel:
    """Map (index, direction) to a semantic ChangeLabel."""
    return _classify_single_index_with_evidence(index_name, mean_delta)[0]


# ---------------------------------------------------------------------------
# Natural-language summary builder
# ---------------------------------------------------------------------------

_DIRECTION_DESCRIPTIONS: dict[str, str] = {
    "vegetation_loss":    "significant vegetation loss was detected",
    "vegetation_gain":    "significant vegetation gain (or regrowth) was detected",
    "water_expansion":    "a substantial expansion of surface water was detected",
    "water_contraction":  "a significant reduction in surface water extent was detected",
    "urban_growth":       "new built-up or impervious surface development was detected",
    "sar_change":         "a change in SAR backscatter consistent with ground-surface alteration was detected",
    "mixed_change":       "conflicting spectral signals suggest mixed land-cover change",
    "no_change":          "no significant change was detected within the analysed area",
}


def build_text_summary(
    area_metrics: dict[str, float],
    direction: ChangeLabel,
    primary_index: str,
    n_regions: int,
    timestamp_t1: str | None = None,
    timestamp_t2: str | None = None,
    n_pseudo_removed: int = 0,
    confidence: float = 0.0,
) -> str:
    """
    Construct a plain-English summary paragraph suitable for the API response
    and the non-expert user UX.

    Returns
    -------
    Multi-sentence summary string.
    """
    direction_desc = _DIRECTION_DESCRIPTIONS.get(direction, "change was detected")
    area_ha = area_metrics.get("area_ha", 0.0)
    pct = area_metrics.get("pct_changed", 0.0)
    n_px = area_metrics.get("n_changed_pixels", 0)

    # Temporal clause
    if timestamp_t1 and timestamp_t2:
        temporal_clause = f"Between {timestamp_t1} and {timestamp_t2}, "
    else:
        temporal_clause = "Between the two input images, "

    # Area clause
    if area_ha >= 1.0:
        area_clause = f"{area_ha:.2f} ha ({pct:.1f}% of the image)"
    else:
        area_clause = f"{int(n_px):,} pixels ({pct:.2f}% of the image)"

    # Region clause
    region_clause = (
        f"spanning {n_regions} distinct region{'s' if n_regions != 1 else ''}"
    )

    # Confidence clause
    conf_clause = f" Confidence score: {confidence:.2f}/1.00."

    # Pseudo-change note
    pseudo_note = ""
    if n_pseudo_removed > 0:
        pseudo_note = (
            f" ({n_pseudo_removed:,} pseudo-change pixels from radiometric drift "
            f"were suppressed before thresholding.)"
        )

    summary = (
        f"{temporal_clause}{direction_desc}, covering approximately {area_clause}, "
        f"{region_clause}. The primary spectral indicator used was "
        f"{primary_index.upper()}.{conf_clause}{pseudo_note}"
    )

    return summary
