"""
Tests for Cross-Modal Arbitration & Calibrated Confidence Decomposition (Critiques 2, 3, 4, 14, 15)
"""

import pytest
from satquery.agent.schemas import SpecialistResult, MeasurementRecord
from satquery.agent.fusion import CrossModalArbitrationEngine, CalibratedConfidenceEngine


def make_specialist_result(tool_id: str, area_ha: float, obs: str, ci_span: float = 0.5) -> SpecialistResult:
    m = MeasurementRecord(
        metric_id="area_ha",
        value=area_ha,
        unit="ha",
        source_tool=tool_id,
        source_run_id="run_test",
        raster_hash="hash_test",
        confidence_interval=(area_ha - ci_span, area_ha + ci_span)
    )
    return SpecialistResult(
        tool_id=tool_id,
        implementation_type="real_model",
        execution_status="healthy",
        duration_ms=50.0,
        observations=obs,
        measurements=[m],
        confidence_score=0.88
    )


def test_concordant_agreement_with_overlapping_intervals():
    opt = make_specialist_result("optical_vlm", 10.2, "Clear optical ground detection", ci_span=0.6)
    sar = make_specialist_result("sar_analyzer", 10.5, "Microwave roughness detection", ci_span=0.6)

    assessment = CrossModalArbitrationEngine.arbitrate(opt, sar)
    assert assessment.category == "concordant_agreement"
    assert assessment.uncertainty_overlap is True
    assert assessment.contradiction_penalty == 0.0


def test_small_area_scaling_prevents_false_contradiction():
    # Relative diff is 50% (1.0 vs 1.5 ha), but absolute is only 0.5 ha!
    opt = make_specialist_result("optical_vlm", 1.0, "Small pond", ci_span=0.1)
    sar = make_specialist_result("sar_analyzer", 1.5, "Small pond backscatter", ci_span=0.1)

    assessment = CrossModalArbitrationEngine.arbitrate(opt, sar)
    # Must NOT be strong contradiction! Small area tolerance triggers.
    assert assessment.category == "concordant_agreement"
    assert assessment.modality_profiles.get("small_area_scaling") is True


def test_cross_modal_cloud_complementarity():
    opt = make_specialist_result("optical_vlm", 5.0, "Heavy cloud cover obscuring visible ground", ci_span=0.2)
    sar = make_specialist_result("sar_analyzer", 18.0, "Microwave penetrates cloud, mapping water", ci_span=0.5)

    assessment = CrossModalArbitrationEngine.arbitrate(opt, sar)
    assert assessment.category == "cross_modal_complementarity"
    assert assessment.contradiction_penalty <= 0.10


def test_strong_contradiction_penalizes_confidence():
    opt = make_specialist_result("optical_vlm", 100.0, "Clear sunny day, detected 100 ha water", ci_span=1.0)
    sar = make_specialist_result("sar_analyzer", 20.0, "Detected 20 ha water", ci_span=1.0)

    assessment = CrossModalArbitrationEngine.arbitrate(opt, sar)
    assert assessment.category == "strong_contradiction"
    assert assessment.contradiction_penalty >= 0.20

    composite, breakdown = CalibratedConfidenceEngine.calculate([opt, sar], assessment)
    assert breakdown.contradiction_penalty >= 0.20
    assert composite < 0.70
