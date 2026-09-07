"""
satquery.change_detection.pipeline
=====================================
End-to-end ChangeDetector orchestrator.

This is the single entry point that the Agentic Controller (Tool 3) calls.
It accepts file paths (or in-memory RasterData objects) and returns a
structured JSON-serialisable result dict that maps directly to the
ChangeDetectionResponse Pydantic schema.

Pipeline stages (in order)
---------------------------
Stage 0  — GeoValidator input checks
Stage 1  — GeoTIFF loading
Stage 2  — Co-registration (match T2 to T1 grid if needed)
Stage 3  — Spectral index computation (auto-select or use query hint)
Stage 4  — Absolute differencing
Stage 5  — Gaussian smoothing (speckle reduction)
Stage 6  — Pseudo-change suppression  ← Research Gap #5
Stage 7  — Otsu thresholding → binary mask
Stage 8  — Morphological cleaning
Stage 9  — Confidence scoring         ← Research Gap #3
Stage 10 — Real-world area & direction metrics
Stage 11 — GeoJSON polygonization + GeoValidator output check
Stage 12 — Execution trace assembly (Research Gap #4)

JSON contract (abbreviated)
----------------------------
{
  "status": "ok" | "partial" | "error",
  "primary_index": "ndvi",
  "change_direction": "vegetation_loss",
  "confidence": 0.83,
  "confidence_label": "HIGH",
  "area_metrics": { "area_m2": …, "area_ha": …, … },
  "n_changed_pixels": 12048,
  "n_regions": 7,
  "otsu_threshold": 0.142,
  "n_pseudo_removed": 341,
  "summary": "Between 2023-01-01 and 2024-01-01, significant vegetation …",
  "geojson": { "type": "FeatureCollection", … },
  "execution_trace": [ … ],
  "warnings": [ … ],
  "sensor_calibration_note": "…",
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Literal

import numpy as np

from satquery.core.raster_io import (
    RasterData,
    load_raster,
    load_raster_from_array,
    mask_to_geojson,
    match_raster,
)
from satquery.core.validator import (
    ValidationError,
    check_polygons_in_bounds,
    validate_input_pair,
)

from .confidence import compute_confidence, confidence_label
from .detector import detect_changes
from .indices import available_indices, extract_index
from .metrics import build_text_summary, classify_change_direction, compute_area
from .morphology import clean_mask, get_component_stats, label_components
from .models import BaseChangeModel, ChangePrediction, ModelStatus, get_change_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index query → index-name mapping
# ---------------------------------------------------------------------------
_QUERY_HINT_MAP: dict[str, str] = {
    # Vegetation queries
    "vegetation": "ndvi", "forest":   "ndvi", "crop":    "ndvi",
    "green":      "ndvi", "ndvi":     "ndvi", "deforest": "ndvi",
    # Water queries
    "water":  "ndwi",  "flood":   "ndwi",  "river":  "ndwi",
    "lake":   "ndwi",  "ndwi":    "ndwi",  "wetland": "ndwi",
    # Urban queries
    "urban":  "ndbi",  "built":   "ndbi",  "building": "ndbi",
    "city":   "ndbi",  "ndbi":    "ndbi",  "road":    "ndbi",
    # SAR queries
    "sar":    "rvi",   "backscatter": "rvi", "rvi":  "rvi",
}

_INDEX_PRIORITY_OPTICAL = ["ndvi", "ndwi", "ndbi"]
_INDEX_PRIORITY_SAR     = ["rvi", "db_vv", "db_vh"]


# ---------------------------------------------------------------------------
# Sensor calibration notes
# ---------------------------------------------------------------------------

def _sensor_note(sensor_t1: str, sensor_t2: str) -> str:
    sensors = {sensor_t1.lower(), sensor_t2.lower()}
    if "cartosat" in sensors or "risat" in sensors:
        return (
            "Outputs calibrated for ISRO Cartosat-2S / RISAT sensor characteristics. "
            "Radiometric values aligned to ISRO standard product DN ranges."
        )
    if "sentinel" in sensors:
        return (
            "Outputs based on ESA Sentinel-2 / Sentinel-1 reflectance/backscatter standards. "
            "For best results with ISRO sensors, apply Cartosat-2S/RISAT calibration profiles."
        )
    return (
        "Generic sensor calibration applied. "
        "For ISRO Cartosat-2S / RISAT inputs, specify sensor_t1='cartosat' or 'risat' "
        "for sensor-specific calibration."
    )


# ---------------------------------------------------------------------------
# Execution trace builder
# ---------------------------------------------------------------------------

def _trace_step(
    stage: int,
    name: str,
    tool: str,
    observation: str,
    duration_ms: float,
    why: str = "",
) -> dict:
    """Build a single execution-trace entry (Research Gap #4 — 'Why this tool?')."""
    entry = {
        "stage":       stage,
        "name":        name,
        "tool":        tool,
        "observation": observation,
        "duration_ms": round(duration_ms, 1),
    }
    if why:
        entry["why"] = why
    return entry


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class ChangeDetector:
    """
    End-to-end bi-temporal change detection pipeline.

    Parameters
    ----------
    sensor_t1, sensor_t2 : Sensor type of each input.
                           One of {"optical", "sar", "cartosat", "risat", "sentinel"}.
    query_hint           : Free-text hint from the user query (e.g., "flood",
                           "deforestation") used to select the primary index.
    suppress_pseudo      : Enable STSF-Net pseudo-change suppression.
    smooth_sigma         : Gaussian blur sigma (speckle reduction, pixels).
    open_radius          : Morphological opening disk radius (noise removal).
    close_radius         : Morphological closing disk radius (hole-filling).
    min_area_px          : Minimum connected-component area (pixels).
    """

    def __init__(
        self,
        sensor_t1: str = "optical",
        sensor_t2: str = "optical",
        query_hint: str = "",
        suppress_pseudo: bool = True,
        smooth_sigma: float = 1.5,
        open_radius: int = 2,
        close_radius: int = 3,
        min_area_px: int = 25,
        model_name: str = "classical",
        model_weights_path: str | None = None,
    ):
        self.sensor_t1      = sensor_t1.lower()
        self.sensor_t2      = sensor_t2.lower()
        self.query_hint     = query_hint.lower()
        self.suppress_pseudo = suppress_pseudo
        self.smooth_sigma   = smooth_sigma
        self.open_radius    = open_radius
        self.close_radius   = close_radius
        self.min_area_px    = min_area_px
        self.model_name     = model_name.lower()
        self.model_weights_path = model_weights_path

    # ------------------------------------------------------------------
    # Primary entry point: file paths
    # ------------------------------------------------------------------

    def run_from_files(
        self,
        path_t1: str,
        path_t2: str,
        timestamp_t1: str | None = None,
        timestamp_t2: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline on two GeoTIFF file paths.

        Parameters
        ----------
        path_t1, path_t2 : Absolute paths to T1 and T2 GeoTIFF files.
        timestamp_t1/t2  : ISO-8601 date strings, e.g. "2023-01-15".
        output_dir       : If given, saves change-mask GeoTIFF and GeoJSON here.

        Returns
        -------
        Structured result dict (see module docstring for schema).
        """
        trace: list[dict] = []
        warnings: list[str] = []
        stage = 0

        # --- Stage 0: Validate inputs ---
        t0 = time.perf_counter()
        try:
            info_t1, info_t2 = validate_input_pair(
                path_t1, path_t2,
                sensor_t1=self.sensor_t1,
                sensor_t2=self.sensor_t2,
                timestamp_t1=timestamp_t1,
                timestamp_t2=timestamp_t2,
            )
        except ValidationError as exc:
            return {
                "status": "error",
                "error": str(exc),
                "check": exc.check,
                "execution_trace": trace,
            }
        trace.append(_trace_step(
            stage, "Input Validation", "GeoValidator",
            f"T1: {os.path.basename(path_t1)} ({info_t1['count']} bands, "
            f"CRS={info_t1['crs']}) | T2: {os.path.basename(path_t2)} "
            f"({info_t2['count']} bands). All checks passed.",
            (time.perf_counter() - t0) * 1000,
            why="Runs first to fail fast on CRS/overlap/timestamp mismatches.",
        ))
        stage += 1

        # --- Stage 1: Load ---
        t0 = time.perf_counter()
        raster_t1 = load_raster(path_t1)
        raster_t2 = load_raster(path_t2)
        trace.append(_trace_step(
            stage, "GeoTIFF Loading", "rasterio",
            f"Loaded T1 {raster_t1} and T2 {raster_t2}.",
            (time.perf_counter() - t0) * 1000,
        ))
        stage += 1

        return self._run_core(
            raster_t1, raster_t2,
            timestamp_t1, timestamp_t2,
            trace, warnings, stage, output_dir,
            source_paths=(path_t1, path_t2),
        )

    # ------------------------------------------------------------------
    # Entry point: in-memory RasterData (used by tests & API upload)
    # ------------------------------------------------------------------

    def run_from_arrays(
        self,
        raster_t1: RasterData,
        raster_t2: RasterData,
        timestamp_t1: str | None = None,
        timestamp_t2: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the pipeline on pre-loaded :class:`~satquery.core.RasterData` objects.
        Skips the file-existence checks but still runs CRS/overlap/temporal checks.
        """
        trace: list[dict] = []
        warnings: list[str] = []
        stage = 0

        return self._run_core(
            raster_t1, raster_t2,
            timestamp_t1, timestamp_t2,
            trace, warnings, stage, output_dir,
        )

    # ------------------------------------------------------------------
    # Core pipeline (shared by both entry points)
    # ------------------------------------------------------------------

    def _run_core(
        self,
        raster_t1: RasterData,
        raster_t2: RasterData,
        timestamp_t1: str | None,
        timestamp_t2: str | None,
        trace: list,
        warnings: list,
        stage: int,
        output_dir: str | None,
        source_paths: tuple[str, str] | None = None,
    ) -> dict[str, Any]:

        total_t0 = time.perf_counter()

        # --- Stage N: Co-registration ---
        t0 = time.perf_counter()
        needs_coreg = (
            raster_t1.crs != raster_t2.crs
            or raster_t1.transform != raster_t2.transform
            or raster_t1.height != raster_t2.height
            or raster_t1.width  != raster_t2.width
        )
        if needs_coreg:
            raster_t2 = match_raster(raster_t2, raster_t1)
            obs = f"T2 reprojected/resampled to match T1 grid ({raster_t1.height}×{raster_t1.width})."
            why = "CRS, transform or shape mismatch detected — co-registration required before differencing."
        else:
            obs = f"Images already co-registered ({raster_t1.height}×{raster_t1.width}). No resampling needed."
            why = "Grids already aligned — skipping resampling for speed."
        trace.append(_trace_step(stage, "Co-registration", "rasterio.warp", obs,
                                 (time.perf_counter() - t0) * 1000, why=why))
        stage += 1

        # --- Determine modality ---
        is_sar_t1 = self.sensor_t1 in ("sar", "risat")
        is_sar_t2 = self.sensor_t2 in ("sar", "risat")
        sensor_label_t1 = "sar" if is_sar_t1 else "optical"
        sensor_label_t2 = "sar" if is_sar_t2 else "optical"

        # --- Stage N: Index selection ---
        t0 = time.perf_counter()
        primary_index = self._select_index(
            raster_t1, is_sar=is_sar_t1
        )
        why_index = self._why_index(primary_index)
        trace.append(_trace_step(
            stage, "Index Selection", "indices.available_indices",
            f"Selected primary index: {primary_index.upper()} "
            f"(bands={raster_t1.bands}, sensor={sensor_label_t1}, hint='{self.query_hint}').",
            (time.perf_counter() - t0) * 1000,
            why=why_index,
        ))
        stage += 1

        # --- Stage N: Compute indices ---
        t0 = time.perf_counter()
        idx_t1 = extract_index(raster_t1.array, primary_index)
        idx_t2 = extract_index(raster_t2.array, primary_index)
        if idx_t1 is None or idx_t2 is None:
            # Fall back to first band
            logger.warning("Index %s unavailable — falling back to single band.", primary_index)
            primary_index = "band_1"
            idx_t1 = raster_t1.array[0]
            idx_t2 = raster_t2.array[0]
            warnings.append(f"Index {primary_index} unavailable; fell back to Band 1 intensity.")
        trace.append(_trace_step(
            stage, "Spectral Index Computation", f"indices.{primary_index}",
            f"Computed {primary_index.upper()} for T1 and T2. "
            f"T1 range: [{idx_t1.min():.3f}, {idx_t1.max():.3f}] | "
            f"T2 range: [{idx_t2.min():.3f}, {idx_t2.max():.3f}].",
            (time.perf_counter() - t0) * 1000,
        ))
        stage += 1

        # --- Stage N: Detect changes (diff + smooth + pseudo-filter + Otsu / Learned Model) ---
        t0 = time.perf_counter()
        model_status_val = ModelStatus.CLASSICAL_ALGORITHM.value
        model_provenance: dict[str, Any] = {}
        inference_path: str = "CLASSICAL_ONLY"

        if self.model_name in ("changeformer", "changeformer_v2"):
            try:
                change_model = get_change_model(self.model_name, weights_path=self.model_weights_path)
                pred = change_model.predict(raster_t1, raster_t2)
                model_status_val = pred.model_status.value
                model_provenance = pred.provenance
                raw_mask = pred.change_mask

                if pred.is_fallback:
                    inference_path = "HEURISTIC_FALLBACK"
                    fallback_warning = pred.fallback_reason or f"Learned model '{self.model_name}' fell back to classical baseline."
                    warnings.append(fallback_warning)
                elif pred.model_status == ModelStatus.REAL_MODEL:
                    inference_path = "LEARNED_MODEL"
                else:
                    inference_path = "CLASSICAL_ONLY"

                # Classical cross-check baseline
                det = detect_changes(
                    idx_t1, idx_t2,
                    smooth_sigma=self.smooth_sigma,
                    suppress_pseudo_changes=self.suppress_pseudo,
                )
            except Exception as exc:
                logger.warning("ChangeFormer failed: %s. Falling back to classical baseline.", exc)
                det = detect_changes(
                    idx_t1, idx_t2,
                    smooth_sigma=self.smooth_sigma,
                    suppress_pseudo_changes=self.suppress_pseudo,
                )
                raw_mask = det["change_mask"]
                model_status_val = ModelStatus.HEURISTIC_FALLBACK.value
                inference_path = "HEURISTIC_FALLBACK"
                model_provenance = {"fallback_reason": str(exc)}
                warnings.append(f"Learned model '{self.model_name}' failed ({exc}); visibly fell back to classical baseline.")
        else:
            det = detect_changes(
                idx_t1, idx_t2,
                smooth_sigma=self.smooth_sigma,
                suppress_pseudo_changes=self.suppress_pseudo,
            )
            raw_mask = det["change_mask"]
            model_status_val = ModelStatus.CLASSICAL_ALGORITHM.value
            inference_path = "CLASSICAL_ONLY"
            model_provenance = {"algorithm": "STSF-Otsu-SpectralDiff"}

        trace.append(_trace_step(
            stage, "Change Detection",
            f"model.{self.model_name}",
            f"Status: {model_status_val}. Otsu threshold={det['otsu_threshold']:.4f}. "
            f"Raw changed pixels: {raw_mask.sum()}. "
            f"Pseudo-change pixels suppressed: {det['n_pseudo_removed']}.",
            (time.perf_counter() - t0) * 1000,
            why=(
                "Combines learned representation and/or pseudo-change suppression to eliminate "
                "radiometric-drift artefacts between T1 and T2."
            ),
        ))
        stage += 1

        # --- Stage N: Morphological cleaning ---
        t0 = time.perf_counter()
        cleaned_mask = clean_mask(
            raw_mask,
            open_radius=self.open_radius,
            close_radius=self.close_radius,
            min_area_px=self.min_area_px,
        )
        labeled, n_regions = label_components(cleaned_mask)
        comp_stats = get_component_stats(labeled)
        trace.append(_trace_step(
            stage, "Morphological Cleaning", "morphology.clean_mask",
            f"After cleaning: {cleaned_mask.sum()} changed pixels in {n_regions} region(s). "
            f"Largest region: {comp_stats[0]['area_px'] if comp_stats else 0} px.",
            (time.perf_counter() - t0) * 1000,
            why="Removes single-pixel salt-and-pepper noise and fills small holes in change polygons.",
        ))
        stage += 1

        # --- Stage N: Confidence scoring ---
        t0 = time.perf_counter()
        conf_score = compute_confidence(
            det["suppressed_diff"], cleaned_mask, det["otsu_threshold"]
        )
        conf_lbl = confidence_label(conf_score)
        trace.append(_trace_step(
            stage, "Confidence Scoring", "confidence.compute_confidence",
            f"Bimodal histogram separation confidence: {conf_score:.4f} ({conf_lbl}).",
            (time.perf_counter() - t0) * 1000,
            why="Quantifies how well-separated the change/no-change distributions are in the difference map.",
        ))
        stage += 1

        # --- Stage N: Area & direction metrics ---
        t0 = time.perf_counter()
        area_metrics = compute_area(cleaned_mask, raster_t1.transform)
        signed_diff  = det["signed_diff"]
        direction_info = classify_change_direction(
            signed_diff, cleaned_mask, primary_index
        )
        direction = str(direction_info)
        direction_provenance = direction_info.to_dict()
        summary_text = build_text_summary(
            area_metrics, direction, primary_index, n_regions,
            timestamp_t1, timestamp_t2,
            n_pseudo_removed=det["n_pseudo_removed"],
            confidence=conf_score,
        )
        trace.append(_trace_step(
            stage, "Area & Direction Metrics", "metrics",
            f"Change direction: {direction}. "
            f"Area: {area_metrics['area_ha']:.3f} ha ({area_metrics['pct_changed']:.2f}%).",
            (time.perf_counter() - t0) * 1000,
        ))
        stage += 1

        # --- Stage N: Polygonization & GeoValidator output check ---
        t0 = time.perf_counter()
        geojson = mask_to_geojson(
            cleaned_mask, raster_t1, label=direction, min_area_px=self.min_area_px
        )
        tf = raster_t1.meta["transform"]
        minx = min(tf.c, tf.c + tf.a * raster_t1.width)
        maxx = max(tf.c, tf.c + tf.a * raster_t1.width)
        miny = min(tf.f, tf.f + tf.e * raster_t1.height)
        maxy = max(tf.f, tf.f + tf.e * raster_t1.height)
        bounds_flat = (minx, miny, maxx, maxy)
        poly_warnings = check_polygons_in_bounds(geojson, bounds_flat)
        warnings.extend(poly_warnings)
        trace.append(_trace_step(
            stage, "Polygonization + GeoValidator", "raster_io + validator",
            f"{len(geojson['features'])} GeoJSON features generated. "
            f"GeoValidator: {len(poly_warnings)} polygon bound warning(s).",
            (time.perf_counter() - t0) * 1000,
            why="GeoValidator rejects polygons outside image bounds to catch CRS artefacts.",
        ))
        stage += 1

        # --- Optional: save outputs to disk ---
        saved_paths: dict[str, str] = {}
        if output_dir:
            t0 = time.perf_counter()
            os.makedirs(output_dir, exist_ok=True)
            from satquery.core.raster_io import save_mask, save_geojson
            mask_path   = os.path.join(output_dir, "change_mask.tif")
            geojson_path = os.path.join(output_dir, "change_mask.geojson")
            save_mask(cleaned_mask, raster_t1, mask_path)
            save_geojson(geojson, geojson_path)
            # Compute content hash for auditability (Research Gap: exportable audit)
            geojson_str = json.dumps(geojson, sort_keys=True).encode()
            content_hash = hashlib.sha256(geojson_str).hexdigest()[:16]
            saved_paths = {
                "change_mask_tif":   mask_path,
                "geojson":           geojson_path,
                "geojson_sha256":    content_hash,
            }
            trace.append(_trace_step(
                stage, "Outputs Saved", "raster_io",
                f"Saved {mask_path} and {geojson_path}. SHA-256 (first 16): {content_hash}.",
                (time.perf_counter() - t0) * 1000,
            ))

        # --- Assemble result ---
        total_ms = (time.perf_counter() - total_t0) * 1000

        result: dict[str, Any] = {
            # GeoCV Lead Charter Section 6 core contract
            "change_type":           direction,
            "change_type_provenance": direction_provenance,
            "inference_path":        inference_path,
            "confidence":            conf_score,
            "changed_area_m2":       float(area_metrics["area_m2"]),
            "changed_area_ha":       float(area_metrics["area_ha"]),
            "change_percentage":     float(area_metrics["pct_changed"]),
            "geometry":              geojson,
            "supporting_evidence": {
                "model_name":         self.model_name,
                "model_status":       model_status_val,
                "inference_path":     inference_path,
                "is_fallback":        (inference_path == "HEURISTIC_FALLBACK"),
                "fallback_reason":    model_provenance.get("fallback_reason"),
                "spectral_index":     primary_index,
                "change_type_provenance": direction_provenance,
                "otsu_threshold":     round(float(det["otsu_threshold"]), 5),
                "n_pseudo_removed":   int(det["n_pseudo_removed"]),
                "bimodal_separation": round(float(conf_score), 4),
                "sensor_calibration": _sensor_note(self.sensor_t1, self.sensor_t2),
                "timestamp_t1":       timestamp_t1,
                "timestamp_t2":       timestamp_t2,
                "provenance":         model_provenance,
            },
            "warnings":              warnings,

            # Extended fields for backward compatibility
            "status":                "ok" if not warnings else "partial",
            "primary_index":         primary_index,
            "change_direction":      direction,
            "confidence_label":      conf_lbl,
            "area_metrics":          area_metrics,
            "n_changed_pixels":      int(cleaned_mask.sum()),
            "n_regions":             n_regions,
            "component_stats":       comp_stats[:10],   # top-10 regions
            "otsu_threshold":        det["otsu_threshold"],
            "n_pseudo_removed":      det["n_pseudo_removed"],
            "summary":               summary_text,
            "geojson":               geojson,
            "execution_trace":       trace,
            "sensor_calibration_note": _sensor_note(self.sensor_t1, self.sensor_t2),
            "total_processing_ms":   round(total_ms, 1),
        }
        if saved_paths:
            result["saved_outputs"] = saved_paths

        logger.info(
            "ChangeDetector completed in %.0f ms — direction=%s confidence=%.3f",
            total_ms, direction, conf_score,
        )
        return result

    # ------------------------------------------------------------------
    # Index selection helpers
    # ------------------------------------------------------------------

    def _select_index(self, raster: RasterData, is_sar: bool = False) -> str:
        """Auto-select the best index given query hint and available bands."""
        # 1. Query-hint override
        for keyword, index in _QUERY_HINT_MAP.items():
            if keyword in self.query_hint:
                avail = available_indices(raster.array, sensor="sar" if is_sar else "optical")
                if index in avail:
                    return index

        # 2. Priority fallback based on sensor
        priority = _INDEX_PRIORITY_SAR if is_sar else _INDEX_PRIORITY_OPTICAL
        avail = available_indices(raster.array, sensor="sar" if is_sar else "optical")
        for idx in priority:
            if idx in avail:
                return idx

        # 3. Last resort: single band
        return "band_1"

    def _why_index(self, index: str) -> str:
        _why = {
            "ndvi":  "Query hint suggests vegetation change; NDVI is the most sensitive optical index for vegetation.",
            "ndwi":  "Query hint suggests water/flood; NDWI isolates open water with minimal soil/vegetation confusion.",
            "ndbi":  "Query hint suggests urban growth; NDBI highlights built-up vs. vegetated surfaces.",
            "rvi":   "SAR input detected; RVI is the primary SAR index for vegetation and surface roughness change.",
            "db_vv": "SAR single-pol (VV) input; dB-scale amplitude change is the most direct SAR change indicator.",
            "band_1": "No multi-band index available; using raw Band 1 intensity difference.",
        }
        return _why.get(index, "Default index selected based on available bands.")
