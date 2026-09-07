"""
Structured Intent Parser & Context Resolver (Workstream E & Platform)
Translates user queries and mission context into typed QueryIntent objects.
Defends against prompt injection by parsing into structured whitelisted attributes.
"""

import re
from typing import Dict, Any, List, Optional
from satquery.agent.schemas import QueryIntent, ContextResolution

# Whitelisted keywords for rule-based matching
INTENT_KEYWORDS = {
    "temporal_change": ["change", "difference", "t1", "t2", "between", "before and after", "evolution", "trend"],
    "deforestation": ["deforestation", "forest loss", "tree cover", "logging", "clearing", "canopy"],
    "flood_mapping": ["flood", "water expansion", "inundation", "submerged", "overflow", "lake spread"],
    "grounding": ["locate", "box", "detect", "bounding box", "find", "where is", "coordinates"],
    "scene_description": ["describe", "caption", "overview", "what is in", "summary of scene", "scene description"],
    "sar_cloud_penetration": ["sar", "radar", "cloud", "smoke", "sentinel-1", "penetrate", "backscatter", "all-weather"],
    "multimodal_fusion": ["fusion", "optical and sar", "fuse", "cross-sensor", "complementary", "disagreement"],
    "vqa": ["how many", "what is", "is there", "count", "type of", "classification"]
}


class IntentParser:
    """
    Parses user natural language query into typed QueryIntent.
    Supports multi-turn context resolution.
    """

    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitizes untrusted input string.
        Strips potential prompt injection patterns (system overrides, markdown injection).
        """
        # Remove system prompt injection tokens
        sanitized = re.sub(r'(?i)(ignore previous instructions|system:|assistant:|<\|im_start\|>|<\|im_end\|>)', '', text)
        return sanitized.strip()

    @classmethod
    def parse_query(
        cls,
        query: str,
        active_assets: Optional[Dict[str, str]] = None,
        prior_context: Optional[Dict[str, Any]] = None
    ) -> QueryIntent:
        clean_query = cls.sanitize_input(query)
        lower_q = clean_query.lower()

        # 1. Determine task_type
        task_type = "vqa"  # Default
        for candidate_type, keywords in INTENT_KEYWORDS.items():
            if any(kw in lower_q for kw in keywords):
                task_type = candidate_type
                break

        # 2. Extract modalities
        modalities = []
        if "sar" in lower_q or "radar" in lower_q or "sentinel-1" in lower_q:
            modalities.append("sar")
        if "optical" in lower_q or "sentinel-2" in lower_q or "rgb" in lower_q:
            modalities.append("optical")
        if not modalities:
            modalities = ["optical"]

        # 3. Temporal flag
        temporal = (
            "change" in lower_q or "t1" in lower_q or "t2" in lower_q or
            "before" in lower_q or "after" in lower_q or
            (active_assets is not None and len(active_assets) > 1)
        )

        # 4. Target entity
        target_entity = "landcover"
        for entity in ["water", "flood", "forest", "urban", "building", "vegetation", "agriculture", "road"]:
            if entity in lower_q:
                target_entity = entity
                break

        # 5. Requested outputs
        requested = ["answer"]
        if "area" in lower_q or "hectare" in lower_q or "size" in lower_q or "how much" in lower_q or task_type in ["temporal_change", "deforestation", "flood_mapping"]:
            requested.append("area")
        if "map" in lower_q or "mask" in lower_q or "geojson" in lower_q or temporal:
            requested.extend(["change_map", "geojson"])
        if "box" in lower_q or "locate" in lower_q or task_type == "grounding":
            requested.append("bounding_boxes")

        # Deduplicate requested outputs
        requested_outputs = list(dict.fromkeys(requested))

        # 6. Multi-turn context resolution
        context_resolutions: List[ContextResolution] = []
        if prior_context and ("same area" in lower_q or "same region" in lower_q or "there" in lower_q):
            last_aoi = prior_context.get("last_aoi_geojson")
            last_asset = prior_context.get("last_asset_id")
            context_resolutions.append(
                ContextResolution(
                    reference_phrase="same area",
                    resolved_asset_id=last_asset,
                    resolved_aoi_geojson=last_aoi,
                    confidence=1.0,
                    justification="Matches preceding mission session AOI bounding geometry."
                )
            )

        return QueryIntent(
            schema_version="0.2",
            task_type=task_type,
            target_entity=target_entity,
            requested_outputs=requested_outputs,
            modalities=modalities,
            temporal=temporal,
            spatial_constraint=context_resolutions[0].resolved_aoi_geojson if context_resolutions else None,
            confidence_requirement="standard",
            context_resolutions=context_resolutions
        )
