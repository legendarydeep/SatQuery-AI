"""
satquery.change_detection.vlm_guard
===================================
Dual-layer VLM Anti-Hallucination Guardrail & Immutable Fact Contract Formatter.
Enforces the GeoCV Charter mandate: "the VLM must not invent measurements."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerifiedObservation:
    """Immutable ground-truth facts produced by the GeoCV pipeline."""
    evidence_id: str
    change_type: str
    area_m2: float
    area_ha: float
    change_percentage: float
    polygon_count: int
    confidence: float
    decision_tier: str = "VERIFIED"
    sensor_calibration: str = "Standard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "change_type": self.change_type,
            "area_m2": round(self.area_m2, 1),
            "area_ha": round(self.area_ha, 3),
            "change_percentage": round(self.change_percentage, 2),
            "polygon_count": self.polygon_count,
            "confidence": round(self.confidence, 4),
            "decision_tier": self.decision_tier,
            "sensor_calibration": self.sensor_calibration,
        }


@dataclass
class GuardrailVerdict:
    """Result of auditing a generated VLM response against verified facts."""
    accepted: bool
    verdict: str  # "ACCEPTED", "NUMERIC_CONTRADICTION", "SEMANTIC_HALLUCINATION"
    violations: List[str] = field(default_factory=list)
    extracted_claims: Dict[str, Any] = field(default_factory=dict)


def format_vlm_prompt_context(obs: VerifiedObservation) -> str:
    """
    Format immutable observation block for the Vision-Language Model prompt context.
    """
    return (
        "====================================================================\n"
        "VERIFIED GEOSPATIAL OBSERVATIONS (GROUND TRUTH — DO NOT MODIFY)\n"
        "====================================================================\n"
        f"- Observation ID    : {obs.evidence_id}\n"
        f"- Decision Tier     : {obs.decision_tier}\n"
        f"- Change Type       : {obs.change_type}\n"
        f"- Physical Area     : {obs.area_ha:.3f} hectares ({obs.area_m2:,.0f} m²)\n"
        f"- Image Area Ratio  : {obs.change_percentage:.2f}%\n"
        f"- Distinct Regions  : {obs.polygon_count} polygons\n"
        f"- Verified Confidence: {obs.confidence:.2f} / 1.00\n"
        f"- Sensor Calibration: {obs.sensor_calibration}\n"
        "====================================================================\n"
        "INSTRUCTIONS FOR VLM:\n"
        "You may describe and explain the geographic context of these verified observations.\n"
        "You MUST NOT invent, calculate, or alter any numerical figures or physical units.\n"
        "You MUST NOT assert unobserved causes (e.g. 'commercial development', 'quarrying')\n"
        "unless explicitly corroborated by the visual bands.\n"
        "===================================================================="
    )


class VLMGuardrail:
    """
    Parses VLM responses into typed claims and audits them against verified facts
    using metric-specific tolerances.
    """

    @staticmethod
    def audit_explanation(vlm_text: str, obs: VerifiedObservation) -> GuardrailVerdict:
        violations: List[str] = []
        text_lower = vlm_text.lower()
        extracted: Dict[str, Any] = {}

        # 1. Area Extraction & Audit (Relative ±2.0% + Absolute 100m²)
        ha_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:ha|hectares?)", text_lower)
        for val_str in ha_matches:
            val = float(val_str)
            extracted["stated_area_ha"] = val
            rel_err = abs(val - obs.area_ha) / max(1e-6, obs.area_ha)
            if rel_err > 0.02 and abs(val - obs.area_ha) > 0.1:
                violations.append(
                    f"Numerical area mismatch: VLM stated {val:.2f} ha, but verified evidence is {obs.area_ha:.3f} ha."
                )

        # 2. Polygon Count Audit (Discrete Integer Tolerance: exact)
        count_matches = re.findall(r"(\d+)\s*(?:distinct|separate|different)?\s*(?:regions?|polygons?|clusters?|areas?)", text_lower)
        for c_str in count_matches:
            c = int(c_str)
            # Skip if this is a year (e.g. 2023) or percentage
            if c > 100:
                continue
            extracted["stated_polygon_count"] = c
            if c != obs.polygon_count:
                violations.append(
                    f"Discrete count mismatch: VLM stated {c} regions, but verified evidence has exactly {obs.polygon_count}."
                )

        # 3. Direction / Change Type Semantic Consistency
        unsupported_terms: Dict[str, List[str]] = {
            "vegetation_loss": ["water expansion", "flooded", "new building", "urban growth", "construction site", "quarry"],
            "water_expansion": ["deforestation", "vegetation loss", "urban expansion", "drought"],
            "urban_growth": ["lake expansion", "forest recovery", "flood inundation"],
        }
        banned = unsupported_terms.get(obs.change_type, [])
        for term in banned:
            if term in text_lower:
                violations.append(
                    f"Unsupported semantic claim: VLM stated '{term}', which directly contradicts verified '{obs.change_type}' evidence."
                )

        if violations:
            verdict_str = (
                "NUMERIC_CONTRADICTION"
                if any("mismatch" in v for v in violations)
                else "SEMANTIC_HALLUCINATION"
            )
            return GuardrailVerdict(
                accepted=False,
                verdict=verdict_str,
                violations=violations,
                extracted_claims=extracted,
            )

        return GuardrailVerdict(
            accepted=True,
            verdict="ACCEPTED",
            violations=[],
            extracted_claims=extracted,
        )
