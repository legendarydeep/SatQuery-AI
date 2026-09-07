"""
Capability-Specific Fallback Policy Engine (Critiques 7 & 8)
Governs how the execution engine degrades gracefully when primary specialists fail or timeout.
Enforces the core scientific rule: NEVER substitute an unrelated algorithm merely to return an answer.
"""

from typing import Dict, List, Optional, Any
from satquery.agent.schemas import SpecialistResult, MeasurementRecord


class CapabilityFallbackRule:
    def __init__(
        self,
        capability: str,
        primary_tool_id: str,
        approved_fallback_chain: List[str],
        allow_null_response: bool = True
    ):
        self.capability = capability
        self.primary_tool_id = primary_tool_id
        self.fallback_chain = approved_fallback_chain
        self.allow_null_response = allow_null_response


# Global capability-specific fallback definitions
CAPABILITY_FALLBACK_RULES: Dict[str, CapabilityFallbackRule] = {
    "temporal_change": CapabilityFallbackRule(
        capability="temporal_change",
        primary_tool_id="changeformer_v2",
        approved_fallback_chain=["classical_change_detector", "spectral_diff_otsu"],
        allow_null_response=False
    ),
    "scene_description": CapabilityFallbackRule(
        capability="scene_description",
        primary_tool_id="geochat_vlm",
        approved_fallback_chain=["compact_cpu_vlm"],
        allow_null_response=True  # Spectral diff is NOT an acceptable fallback for scene description!
    ),
    "vqa": CapabilityFallbackRule(
        capability="vqa",
        primary_tool_id="geochat_vlm",
        approved_fallback_chain=["compact_cpu_vlm"],
        allow_null_response=True
    ),
    "grounding": CapabilityFallbackRule(
        capability="grounding",
        primary_tool_id="geochat_grounding",
        approved_fallback_chain=["spectral_clusterer_heuristic"],
        allow_null_response=True
    ),
    "flood_mapping": CapabilityFallbackRule(
        capability="flood_mapping",
        primary_tool_id="multimodal_optical_sar_fusion",
        approved_fallback_chain=["sar_backscatter_otsu", "optical_ndwi_threshold"],
        allow_null_response=False
    ),
    "deforestation": CapabilityFallbackRule(
        capability="deforestation",
        primary_tool_id="changeformer_v2",
        approved_fallback_chain=["ndvi_bitemporal_differencing"],
        allow_null_response=False
    ),
    "sar_cloud_penetration": CapabilityFallbackRule(
        capability="sar_cloud_penetration",
        primary_tool_id="sar_microwave_analyzer",
        approved_fallback_chain=["classical_sar_backscatter"],
        allow_null_response=True
    ),
    "multimodal_fusion": CapabilityFallbackRule(
        capability="multimodal_fusion",
        primary_tool_id="multimodal_optical_sar_fusion",
        approved_fallback_chain=["optical_solo_pipeline", "sar_solo_pipeline"],
        allow_null_response=False
    ),
}


class FallbackPolicyEngine:
    """
    Evaluates specialist failure and selects scientifically valid alternative.
    """

    @staticmethod
    def get_fallback_tool(capability: str, failed_tool_id: str) -> Optional[str]:
        """
        Returns next approved fallback tool for capability, or None if no valid fallback exists.
        """
        rule = CAPABILITY_FALLBACK_RULES.get(capability)
        if not rule:
            return None

        chain = [rule.primary_tool_id] + rule.fallback_chain
        try:
            current_idx = chain.index(failed_tool_id)
            if current_idx + 1 < len(chain):
                return chain[current_idx + 1]
        except ValueError:
            # Failed tool was not in chain; return first available fallback
            if rule.fallback_chain:
                return rule.fallback_chain[0]

        return None

    @staticmethod
    def create_unavailable_result(
        capability: str,
        failed_tool_id: str,
        reason: str
    ) -> SpecialistResult:
        """
        Creates an honest 'unavailable' result rather than fabricating an answer.
        """
        return SpecialistResult(
            tool_id=failed_tool_id,
            implementation_type="heuristic",
            execution_status="unavailable",
            duration_ms=0.0,
            observations=f"Service unavailable for capability '{capability}'. Reason: {reason}",
            measurements=[],
            geojson_geometry=None,
            confidence_score=0.0,
            warnings=[f"Execution terminated honestly: No validated fallback exists for {capability}."]
        )
