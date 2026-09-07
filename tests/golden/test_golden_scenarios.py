import json
import pytest
from pathlib import Path
from satquery.evaluation.schemas import EvaluationRecord, ConfidenceBreakdown
from satquery.evaluation.adapters import (
    ChangeDetectionAdapter,
    AgentPlannerAdapter,
    GenericSpecialistAdapter,
)
from satquery.evaluation.guardrails import NumericalGuardrail

CACHE_DIR = Path("demo/fallback_cache")


@pytest.fixture
def raw_deforestation():
    return {
        "status": "classical_algorithm",
        "summary": "14.20 ha of deforestation detected with 11.50 percent change.",
        "confidence": 0.884,
        "area_metrics": {"area_ha": 14.20, "area_m2": 142000.0, "pct_changed": 11.50},
        "execution_trace": [{"tool": "ndvi_diff"}, {"tool": "otsu"}],
    }


@pytest.fixture
def raw_flood():
    return {
        "status": "classical_algorithm",
        "summary": "92.3 ha flooding confirmed with SAR amplitude increase of 4.1 dB.",
        "confidence": 0.912,
        "area_metrics": {"area_ha": 92.3, "area_m2": 923000.0, "pct_changed": 34.1},
        "execution_trace": [{"tool": "sar_amplitude"}, {"tool": "ndwi"}],
    }


@pytest.fixture
def raw_agent_output():
    return {
        "status": "real_model",
        "response_text": "Settlement boundary expanded by 3.2 ha.",
        "confidence": 0.877,
        "confidence_breakdown": {
            "input_quality": 0.92,
            "model_confidence": 0.88,
            "evidence_agreement": 0.91,
            "geospatial_validity": 0.95,
            "temporal_validity": 0.97,
            "contradiction_penalty": 0.0,
        },
        "deterministic_numbers": {"area_ha": 3.2, "change_pct": 18.5},
        "tool_sequence": ["intent_parser", "change_detector", "fusion_engine"],
        "latency_ms": 3210.0,
    }


class TestSchemaValidation:
    def test_evaluation_record_all_required_fields(self, raw_deforestation):
        rec = ChangeDetectionAdapter.parse(raw_deforestation, "GOLDEN-01")
        assert rec.task_id == "GOLDEN-01"
        assert rec.schema_version == "0.2"
        assert rec.status in (
            "classical_algorithm", "real_model", "heuristic_fallback",
            "failed", "unavailable",
        )
        assert 0.0 <= rec.confidence_score <= 1.0
        assert rec.confidence_breakdown is not None
        assert isinstance(rec.tool_trace, list)

    def test_agent_record_all_required_fields(self, raw_agent_output):
        rec = AgentPlannerAdapter.parse(raw_agent_output, "GOLDEN-02")
        assert rec.schema_version == "0.2"
        assert rec.status == "real_model"
        assert rec.confidence_score == pytest.approx(0.877)
        assert rec.confidence_breakdown.input_quality == pytest.approx(0.92)

    def test_confidence_breakdown_composite_in_range(self, raw_deforestation):
        rec = ChangeDetectionAdapter.parse(raw_deforestation, "GOLDEN-03")
        composite = rec.confidence_breakdown.calculate_composite()
        assert 0.0 <= composite <= 1.0, f"Composite {composite} out of [0,1]"


class TestNumericalIntegrity:
    def test_area_ha_extracted_correctly(self, raw_deforestation):
        rec = ChangeDetectionAdapter.parse(raw_deforestation, "GOLDEN-04")
        assert rec.deterministic_numbers["area_ha"] == pytest.approx(14.20)

    def test_area_m2_extracted_correctly(self, raw_flood):
        rec = ChangeDetectionAdapter.parse(raw_flood, "GOLDEN-05")
        assert rec.deterministic_numbers["area_m2"] == pytest.approx(923000.0)

    def test_guardrail_accepts_correct_text(self, raw_deforestation):
        guardrail = NumericalGuardrail(tolerance=0.05)
        text = "Total deforested area is 14.20 ha across the region."
        passed, violations, _ = guardrail.verify(text, {"area_ha": 14.20})
        assert passed

    def test_guardrail_flags_drift(self):
        guardrail = NumericalGuardrail(tolerance=0.05)
        passed, violations, _ = guardrail.verify(
            "The area is 20.0 ha.", {"area_ha": 14.20}
        )
        assert not passed
        assert len(violations) >= 1


class TestAdapterRouting:
    def test_change_detection_tool_trace(self, raw_deforestation):
        rec = ChangeDetectionAdapter.parse(raw_deforestation, "GOLDEN-07")
        assert "ndvi_diff" in rec.tool_trace or "otsu" in rec.tool_trace

    def test_agent_planner_tool_sequence(self, raw_agent_output):
        rec = AgentPlannerAdapter.parse(raw_agent_output, "GOLDEN-08")
        assert "fusion_engine" in rec.tool_trace

    def test_generic_specialist_adapter(self):
        raw = {
            "status": "real_model",
            "answer": "Settlement area: 3.2 ha.",
            "confidence": 0.79,
            "numbers": {"area_ha": 3.2},
        }
        rec = GenericSpecialistAdapter.parse(raw, "GOLDEN-09")
        assert rec.confidence_score == pytest.approx(0.79)
        assert rec.deterministic_numbers["area_ha"] == pytest.approx(3.2)


class TestCachedPayloadRoundTrip:
    def test_act2_cache_is_schema_valid(self):
        fpath = CACHE_DIR / "act2_bitemporal_change.json"
        with fpath.open() as f:
            payload = json.load(f)
        breakdown = ConfidenceBreakdown(**payload["confidence_breakdown"])
        rec = EvaluationRecord(
            task_id=payload["task_id"],
            schema_version=payload["schema_version"],
            status=payload["status"],
            text_response=payload["text_response"],
            deterministic_numbers=payload["deterministic_numbers"],
            geojson_geometry=payload.get("geojson_geometry"),
            confidence_score=payload["confidence_score"],
            confidence_breakdown=breakdown,
            execution_time_ms=payload.get("execution_time_ms", 115.0),
            tool_trace=payload.get("tool_trace", []),
        )
        assert rec.task_id == "DEMO-ACT2-TEMPORAL-CHANGE"
        assert rec.deterministic_numbers["area_ha"] == pytest.approx(14.20)
        assert rec.confidence_breakdown.calculate_composite() >= 0.80
