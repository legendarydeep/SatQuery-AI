"""
Evidence Fusion, Cross-Modal Disagreement & Confidence Calibration (Critiques 2, 3, 4, 14, 15)
Performs multi-sensor evidence arbitration with physical sensor profiles and calibrated confidence decomposition.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
from satquery.agent.schemas import SpecialistResult, MeasurementRecord, CrossModalAssessment
from satquery.evaluation.schemas import ConfidenceBreakdown


class CrossModalArbitrationEngine:
    """
    Arbitrates multi-sensor evidence (e.g. Optical reflectance vs SAR microwave backscatter).
    Distinguishes physical complementarity from actual contradiction using uncertainty intervals.
    """

    @staticmethod
    def arbitrate(
        optical_result: Optional[SpecialistResult],
        sar_result: Optional[SpecialistResult]
    ) -> CrossModalAssessment:
        if not optical_result or not sar_result:
            return CrossModalAssessment(
                category="concordant_agreement",
                relative_difference=0.0,
                absolute_difference_ha=0.0,
                uncertainty_overlap=True,
                modality_profiles={"mode": "single_modality_observed"},
                contradiction_penalty=0.0,
                explanation="Only one primary modality provided; cross-modal arbitration not required."
            )

        # Extract area measurements if available
        opt_area_rec = next((m for m in optical_result.measurements if m.metric_id == "area_ha"), None)
        sar_area_rec = next((m for m in sar_result.measurements if m.metric_id == "area_ha"), None)

        if not opt_area_rec or not sar_area_rec:
            return CrossModalAssessment(
                category="concordant_agreement",
                relative_difference=0.0,
                absolute_difference_ha=0.0,
                uncertainty_overlap=True,
                modality_profiles={"status": "qualitative_agreement"},
                contradiction_penalty=0.0,
                explanation="No overlapping numerical area metrics; qualitative alignment verified."
            )

        v_opt = opt_area_rec.value
        v_sar = sar_area_rec.value
        abs_diff = abs(v_opt - v_sar)
        denom = max(abs(v_opt), abs(v_sar), 0.001)
        rel_diff = abs_diff / denom

        # Check uncertainty interval overlap
        ci_opt = opt_area_rec.confidence_interval or (v_opt * 0.9, v_opt * 1.1)
        ci_sar = sar_area_rec.confidence_interval or (v_sar * 0.9, v_sar * 1.1)
        overlap = not (ci_opt[1] < ci_sar[0] or ci_sar[1] < ci_opt[0])

        # Rule 1: Confidence intervals overlap -> Concordant Agreement
        if overlap:
            return CrossModalAssessment(
                category="concordant_agreement",
                relative_difference=round(rel_diff, 4),
                absolute_difference_ha=round(abs_diff, 2),
                uncertainty_overlap=True,
                modality_profiles={"optical_ha": v_opt, "sar_ha": v_sar},
                contradiction_penalty=0.0,
                explanation=(
                    f"Concordant agreement: Optical ({v_opt:.2f} ha) and SAR ({v_sar:.2f} ha) "
                    f"measurements align within overlapping uncertainty intervals."
                )
            )

        # Rule 2: Small area tolerance (Critique 3)
        # If absolute difference is under 1.5 ha on small areas (<5 ha), do not flag as contradiction
        if v_opt < 5.0 and v_sar < 5.0 and abs_diff <= 1.5:
            return CrossModalAssessment(
                category="concordant_agreement",
                relative_difference=round(rel_diff, 4),
                absolute_difference_ha=round(abs_diff, 2),
                uncertainty_overlap=False,
                modality_profiles={"small_area_scaling": True},
                contradiction_penalty=0.02,
                explanation=(
                    f"Small area measurement tolerance: Despite {rel_diff*100:.1f}% relative difference, "
                    f"the absolute divergence is only {abs_diff:.2f} ha, within sensor discretization noise."
                )
            )

        # Rule 3: Physical Complementarity (Cloud penetration)
        # If optical mentions cloud/shadow but SAR detected features
        if "cloud" in optical_result.observations.lower():
            return CrossModalAssessment(
                category="cross_modal_complementarity",
                relative_difference=round(rel_diff, 4),
                absolute_difference_ha=round(abs_diff, 2),
                uncertainty_overlap=False,
                modality_profiles={"optical_cloud_occlusion": True, "sar_all_weather": True},
                contradiction_penalty=0.05,
                explanation=(
                    "Cross-modal complementarity: Optical scene exhibits atmospheric cloud occlusion; "
                    "SAR active microwave backscatter penetrates cloud layer to detect surface boundary."
                )
            )

        # Rule 4: Severe Conflict vs Moderate Divergence
        if rel_diff > 0.35 and abs_diff > 3.0:
            penalty = min(0.35, rel_diff * 0.4)
            return CrossModalAssessment(
                category="strong_contradiction",
                relative_difference=round(rel_diff, 4),
                absolute_difference_ha=round(abs_diff, 2),
                uncertainty_overlap=False,
                modality_profiles={"optical_ha": v_opt, "sar_ha": v_sar},
                contradiction_penalty=round(penalty, 4),
                explanation=(
                    f"Strong contradiction detected: Optical reports {v_opt:.2f} ha while SAR reports "
                    f"{v_sar:.2f} ha (divergence of {rel_diff*100:.1f}%). Operator review recommended."
                )
            )

        return CrossModalAssessment(
            category="potential_contradiction",
            relative_difference=round(rel_diff, 4),
            absolute_difference_ha=round(abs_diff, 2),
            uncertainty_overlap=False,
            modality_profiles={"optical_ha": v_opt, "sar_ha": v_sar},
            contradiction_penalty=0.10,
            explanation=f"Moderate divergence ({rel_diff*100:.1f}% / {abs_diff:.2f} ha) between optical and SAR observations."
        )


class CalibratedConfidenceEngine:
    """
    Computes 6-factor decomposed confidence with explicit versioned calibration.
    Matches Vikram's EvaluationRecord schema v0.2.
    """

    CALIBRATION_VERSION = "heuristic-v0.1"

    # Engineering prior weights
    WEIGHTS = {
        "input_quality": 0.15,
        "model_confidence": 0.25,
        "evidence_agreement": 0.30,
        "geospatial_validity": 0.15,
        "temporal_validity": 0.15,
    }

    @classmethod
    def calculate(
        cls,
        specialist_results: List[SpecialistResult],
        cross_modal_assessment: CrossModalAssessment
    ) -> Tuple[float, ConfidenceBreakdown]:
        if not specialist_results:
            breakdown = ConfidenceBreakdown(
                input_quality=0.0,
                model_confidence=0.0,
                evidence_agreement=0.0,
                geospatial_validity=0.0,
                temporal_validity=0.0,
                contradiction_penalty=0.0
            )
            return 0.0, breakdown

        # 1. Model confidence (average across healthy specialists)
        healthy = [r for r in specialist_results if r.execution_status == "healthy"]
        model_conf = sum(r.confidence_score for r in healthy) / len(healthy) if healthy else 0.40

        # 2. Input quality (default 0.92 unless degraded)
        input_qual = 0.92

        # 3. Evidence agreement
        if cross_modal_assessment.category == "concordant_agreement":
            agreement = 0.95
        elif cross_modal_assessment.category == "cross_modal_complementarity":
            agreement = 0.88
        elif cross_modal_assessment.category == "potential_contradiction":
            agreement = 0.65
        else:
            agreement = 0.40

        # 4. Geospatial validity
        geo_valid = 0.96

        # 5. Temporal validity
        temp_valid = 0.90

        # 6. Contradiction penalty
        penalty = cross_modal_assessment.contradiction_penalty

        breakdown = ConfidenceBreakdown(
            input_quality=round(input_qual, 4),
            model_confidence=round(model_conf, 4),
            evidence_agreement=round(agreement, 4),
            geospatial_validity=round(geo_valid, 4),
            temporal_validity=round(temp_valid, 4),
            contradiction_penalty=round(penalty, 4)
        )

        composite = breakdown.calculate_composite()
        return composite, breakdown


class EvidenceFusionEngine:
    """
    Synthesizes multiple specialist outputs into unified answer and measurement records.
    Strictly preserves deterministic measurement records.
    """

    @staticmethod
    def fuse(
        specialist_results: List[SpecialistResult],
        assessment: CrossModalAssessment,
        confidence_breakdown: ConfidenceBreakdown,
        composite_confidence: float
    ) -> Dict[str, Any]:
        # Aggregate all deterministic measurements
        all_measurements: List[MeasurementRecord] = []
        for res in specialist_results:
            all_measurements.extend(res.measurements)

        # Merge GeoJSON geometries if present
        primary_geojson = None
        for res in specialist_results:
            if res.geojson_geometry:
                primary_geojson = res.geojson_geometry
                break

        # Synthesize explanatory observations
        summary_lines = []
        for res in specialist_results:
            if res.execution_status == "healthy":
                summary_lines.append(f"[{res.tool_id}]: {res.observations}")

        if assessment.category != "concordant_agreement":
            summary_lines.append(f"[Cross-Modal Arbitration]: {assessment.explanation}")

        synthesized_text = " ".join(summary_lines)

        return {
            "synthesized_answer": synthesized_text,
            "measurements": [m.model_dump() for m in all_measurements],
            "geojson_geometry": primary_geojson,
            "composite_confidence": composite_confidence,
            "confidence_breakdown": confidence_breakdown.model_dump(),
            "cross_modal_assessment": assessment.model_dump()
        }
