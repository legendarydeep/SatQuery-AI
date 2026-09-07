"""
tests/test_scientific_geocv.py
==============================
Scientific sanity, geodetic integrity, and VLM guardrail validation suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from satquery.core.raster_io import RasterData
from satquery.core.geodetic import (
    CRSSelector,
    compute_alignment_quality,
    normalize_pair_geodetic,
)
from satquery.core.band_mapper import BandMapper
from satquery.change_detection.models.base import (
    DecisionTier,
    ModelArtifactStatus,
    RuntimeStatus,
    ValidationStatus,
)
from satquery.change_detection.models.changeformer import ChangeFormerAdapter
from satquery.change_detection.models.classical_adapter import ClassicalSpectralAdapter
from satquery.change_detection.evidence_fusion import EvidenceFusionEngine, EvidenceSignal
from satquery.change_detection.thresholding import ThresholdConfig, ThresholdSelector
from satquery.change_detection.vlm_guard import (
    GuardrailVerdict,
    VerifiedObservation,
    VLMGuardrail,
    format_vlm_prompt_context,
)


def _make_test_raster(h: int = 64, w: int = 64, val: float = 0.5) -> RasterData:
    arr = np.full((4, h, w), val, dtype=np.float32)
    tf = from_bounds(10.0, 20.0, 10.1, 20.1, w, h)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": w,
        "height": h,
        "count": 4,
        "crs": CRS.from_epsg(32643),
        "transform": tf,
    }
    return RasterData(arr, meta)


# ---------------------------------------------------------------------------
# Test 1: Null Baseline (T1 ≈ T2)
# ---------------------------------------------------------------------------

def test_null_baseline():
    """When T1 ≈ T2, evidence fusion must classify as REJECTED / Null Baseline."""
    r1 = _make_test_raster(val=0.5)
    r2 = _make_test_raster(val=0.5001)

    adapter = ClassicalSpectralAdapter()
    pred = adapter.predict(r1, r2)
    
    report = EvidenceFusionEngine.evaluate(
        pred.probability_map,
        pred.change_mask,
        pred.change_mask,
        {"ndvi": np.abs(r2.band(0) - r1.band(0))},
        alignment_quality=1.0,
    )
    assert report.decision_tier == DecisionTier.REJECTED
    assert "Null baseline" in report.diagnostic_reason


# ---------------------------------------------------------------------------
# Test 2: Injected Polygon Perturbation Sanity
# ---------------------------------------------------------------------------

def test_injected_polygon_sanity():
    """Controlled perturbation injection sanity test."""
    # T1: low NDVI (Red=0.8, NIR=0.2) — bare soil everywhere
    arr_t1 = np.zeros((4, 64, 64), dtype=np.float32)
    arr_t1[0] = 0.05   # Blue
    arr_t1[1] = 0.10   # Green
    arr_t1[2] = 0.80   # Red — high → NDVI ≈ -0.6
    arr_t1[3] = 0.20   # NIR

    # T2: high NDVI everywhere EXCEPT central 20x20 stays bare
    # (invert: make the background change, so the patch is "no change")
    # Actually: make central 20x20 vegetation gain (NDVI from -0.6 → +0.77)
    arr_t2 = arr_t1.copy()
    arr_t2[2, 20:40, 20:40] = 0.10   # low Red → high NDVI
    arr_t2[3, 20:40, 20:40] = 0.80   # high NIR → high NDVI

    tf = from_bounds(10.0, 20.0, 10.1, 20.1, 64, 64)
    meta = {
        "driver": "GTiff", "dtype": "float32", "width": 64, "height": 64,
        "count": 4, "crs": CRS.from_epsg(32643), "transform": tf,
    }
    r1 = RasterData(arr_t1, meta)
    r2 = RasterData(arr_t2, dict(meta))

    # Disable pseudo-change suppression — this is a real planted change, not radiometric drift
    adapter = ClassicalSpectralAdapter(default_index="ndvi")
    pred = adapter.predict(r1, r2, suppress_pseudo=False)

    # At least 50% of the overall change mask must be inside the planted patch
    total_changed = int(pred.change_mask.sum())
    inside_patch = int(pred.change_mask[20:40, 20:40].sum())
    assert total_changed > 0, "Detector found no change — check spectral contrast"
    inside_fraction = inside_patch / total_changed
    assert inside_fraction > 0.50, (
        f"Only {inside_fraction:.1%} of changed pixels are inside the planted patch. "
        "Detector is reporting spurious changes elsewhere."
    )


# ---------------------------------------------------------------------------
# Test 3: Spatial Misregistration Warning
# ---------------------------------------------------------------------------

def test_spatial_misregistration_warning():
    """Residual shift of 3 pixels must drop alignment score below 0.70."""
    b1 = np.random.RandomState(42).randn(128, 128)
    b2 = np.roll(b1, shift=(3, 3), axis=(0, 1))

    shift, quality = compute_alignment_quality(b1, b2)
    assert shift >= 2.5
    assert quality < 0.70


# ---------------------------------------------------------------------------
# Test 4: Equal-Area CRS Selection & Distortion Bound
# ---------------------------------------------------------------------------

def test_equal_area_crs_selection():
    """Broad scene extent must trigger equal-area projection selection."""
    broad_bounds = (10.0, 10.0, 15.0, 15.0)  # 5 degrees broad
    crs, is_ea, dist_bound = CRSSelector.select_optimal_crs(
        broad_bounds, CRS.from_epsg(4326), prefer_equal_area=True
    )
    assert is_ea is True
    assert dist_bound <= 0.05  # Area distortion strictly bounded below 0.05%


# ---------------------------------------------------------------------------
# Test 5: GSD Normalization Policy & Scale Warning
# ---------------------------------------------------------------------------

def test_gsd_normalization_scale_warning():
    """GSD ratio > 1.5x must emit scale warning and record native GSDs."""
    r1 = _make_test_raster(val=0.5)
    r2 = _make_test_raster(val=0.5)
    # Simulate 5m vs 10m
    r2.meta["transform"] = from_bounds(10.0, 20.0, 10.2, 20.2, 64, 64)

    _, _, report = normalize_pair_geodetic(r1, r2)
    assert report.scale_ratio >= 1.5
    assert report.scale_warning is True


# ---------------------------------------------------------------------------
# Test 6: Transparent Fallback Logging
# ---------------------------------------------------------------------------

def test_transparent_fallback_logging():
    """Missing weights must explicitly mark is_fallback=True and UNAVAILABLE."""
    adapter = ChangeFormerAdapter(weights_path="/missing/weights.pth", fallback_on_missing=True)
    r1 = _make_test_raster(val=0.2)
    r2 = _make_test_raster(val=0.8)

    pred = adapter.predict(r1, r2)
    assert pred.is_fallback is True
    assert pred.artifact_status == ModelArtifactStatus.UNAVAILABLE
    assert pred.runtime_status == RuntimeStatus.INCOMPATIBLE
    assert "fell back" in pred.fallback_reason


# ---------------------------------------------------------------------------
# Test 7: Discrete VLM Count Hallucination Rejection
# ---------------------------------------------------------------------------

def test_vlm_discrete_count_hallucination_rejected():
    """VLM claiming 4 separate regions when evidence has 3 must be REJECTED."""
    obs = VerifiedObservation(
        evidence_id="OBS_001",
        change_type="vegetation_loss",
        area_m2=12000.0,
        area_ha=1.2,
        change_percentage=3.5,
        polygon_count=3,
        confidence=0.92,
    )

    hallucinated_text = "Analysis shows vegetation loss spanning 4 distinct regions covering 1.2 ha."
    verdict = VLMGuardrail.audit_explanation(hallucinated_text, obs)
    assert verdict.accepted is False
    assert verdict.verdict == "NUMERIC_CONTRADICTION"
    violations = verdict.violations
    assert any("Discrete count mismatch" in v for v in violations)


# ---------------------------------------------------------------------------
# Test 8: Semantic Claim Hallucination Rejection
# ---------------------------------------------------------------------------

def test_vlm_semantic_claim_hallucination_rejected():
    """VLM claiming new building / construction site on vegetation loss must be REJECTED."""
    obs = VerifiedObservation(
        evidence_id="OBS_002",
        change_type="vegetation_loss",
        area_m2=12000.0,
        area_ha=1.2,
        change_percentage=3.5,
        polygon_count=3,
        confidence=0.92,
    )

    unsupported_text = "The forest was cleared for a new construction site covering 1.2 ha."
    verdict = VLMGuardrail.audit_explanation(unsupported_text, obs)
    assert verdict.accepted is False
    assert verdict.verdict == "SEMANTIC_HALLUCINATION"
    assert any("Unsupported semantic claim" in v for v in verdict.violations)


# ---------------------------------------------------------------------------
# Test 9: Valid VLM Explanation Accepted
# ---------------------------------------------------------------------------

def test_vlm_valid_explanation_accepted():
    """Accurate VLM explanation adhering strictly to facts must be ACCEPTED."""
    obs = VerifiedObservation(
        evidence_id="OBS_003",
        change_type="vegetation_loss",
        area_m2=12000.0,
        area_ha=1.2,
        change_percentage=3.5,
        polygon_count=3,
        confidence=0.92,
    )

    valid_text = "Between the two dates, vegetation loss occurred across 3 distinct regions totaling 1.2 ha."
    verdict = VLMGuardrail.audit_explanation(valid_text, obs)
    assert verdict.accepted is True
    assert verdict.verdict == "ACCEPTED"
    assert len(verdict.violations) == 0


# ---------------------------------------------------------------------------
# Test 10: Adaptive Thresholding Unimodal Diagnostic
# ---------------------------------------------------------------------------

def test_adaptive_threshold_unimodal_handling():
    """98% unchanged distribution must select adaptive_tail, avoiding Otsu failure."""
    # 98% zeros, 2% changes
    flat = np.zeros((100, 100), dtype=np.float32)
    flat[45:50, 45:50] = 0.85

    mask, report = ThresholdSelector.select_threshold(flat, ThresholdConfig())
    assert report.method_selected == "adaptive_tail"
    assert report.is_bimodal is False
    assert mask[47, 47] == True  # Recover change


# ---------------------------------------------------------------------------
# Test 11: Evidence Fusion Named Signal: SUPPORT -> VERIFIED
# ---------------------------------------------------------------------------

def test_evidence_fusion_support_signal_verified():
    """High agreement between learned mask and spectral shifts must produce SUPPORT / VERIFIED."""
    mask_learned = np.zeros((64, 64), dtype=bool)
    mask_learned[20:40, 20:40] = True

    mask_classical = np.zeros((64, 64), dtype=bool)
    mask_classical[22:38, 22:38] = True  # Significant overlap

    prob_learned = np.zeros((64, 64), dtype=np.float32)
    prob_learned[20:40, 20:40] = 0.88

    diff_ndvi = np.zeros((64, 64), dtype=np.float32)
    diff_ndvi[20:40, 20:40] = 0.45  # Strong NDVI shift

    report = EvidenceFusionEngine.evaluate(
        prob_learned=prob_learned,
        mask_learned=mask_learned,
        mask_classical=mask_classical,
        spectral_diffs={"ndvi": diff_ndvi},
        alignment_quality=0.95,
    )

    assert report.evidence_signal == EvidenceSignal.SUPPORT
    assert report.decision_tier == DecisionTier.VERIFIED
    assert report.spectral_support_fraction >= 0.70
    assert report.spectral_disagreement_fraction == 0.0
    assert report.pixel_iou > 0.35


# ---------------------------------------------------------------------------
# Test 12: Evidence Fusion Named Signal: DISAGREEMENT -> UNCERTAIN
# ---------------------------------------------------------------------------

def test_evidence_fusion_disagreement_signal_uncertain():
    """Learned change with high confidence contradicted by zero spectral shift must flag DISAGREEMENT / UNCERTAIN."""
    mask_learned = np.zeros((64, 64), dtype=bool)
    mask_learned[20:40, 20:40] = True

    mask_classical = np.zeros((64, 64), dtype=bool)

    prob_learned = np.zeros((64, 64), dtype=np.float32)
    prob_learned[20:40, 20:40] = 0.92  # High model confidence

    # Zero physical change in spectral bands
    diff_ndvi = np.zeros((64, 64), dtype=np.float32)
    diff_ndwi = np.zeros((64, 64), dtype=np.float32)

    report = EvidenceFusionEngine.evaluate(
        prob_learned=prob_learned,
        mask_learned=mask_learned,
        mask_classical=mask_classical,
        spectral_diffs={"ndvi": diff_ndvi, "ndwi": diff_ndwi},
        alignment_quality=0.92,
    )

    assert report.evidence_signal == EvidenceSignal.DISAGREEMENT
    assert report.decision_tier == DecisionTier.UNCERTAIN
    assert report.spectral_disagreement_fraction > 0.35
    assert "Flagged for review" in report.diagnostic_reason
